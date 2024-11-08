import argparse
import cv2
import numpy as np
import tensorrt as trt
import pyzed.sl as sl
import sys
import common
import common_runtime

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4
MODEL_INPUT_SIZE = (640, 640)

def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def normalize(im):
    im = np.asarray(im, dtype="float32")
    im /= 255.0
    im = im.transpose(2, 0, 1)
    im = np.expand_dims(im, axis=0)
    return im.astype("float32")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of TensorRT model.", required=True)
    parser.add_argument('--input_svo_file', type=str, required=True, help='Path to the .svo file')
    parser.add_argument('--resolution', type=str, help='Resolution: HD2K, HD1200, HD1080, HD720, SVGA, VGA',
                        default='HD720')
    return parser.parse_args()


def setup_zed_camera(opt):
    init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA,
                             coordinate_units=sl.UNIT.MILLIMETER,
                             coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)

    # Set input SVO file and resolution
    init.set_from_svo_file(opt.input_svo_file)
    print(f"[INFO] Using SVO file input: {opt.input_svo_file}")

    resolution_dict = {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1200": sl.RESOLUTION.HD1200,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "SVGA": sl.RESOLUTION.SVGA,
        "VGA": sl.RESOLUTION.VGA
    }
    init.camera_resolution = resolution_dict.get(opt.resolution.upper(), sl.RESOLUTION.HD720)
    print(f"[INFO] Camera resolution set to {opt.resolution.upper()}")

    zed = sl.Camera()
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("Error opening ZED Camera")
        zed.close()
        sys.exit(1)
    return zed


def main():
    args = parse_args()

    yolo_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
        "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    # Load TensorRT model
    engine = get_engine(args.model)
    context = engine.create_execution_context()

    # Initialize ZED camera from SVO
    zed = setup_zed_camera(args)
    runtime_parameters = sl.RuntimeParameters()
    depth_map = sl.Mat()
    left_image = sl.Mat()

    nb_frames = zed.get_svo_number_of_frames()

    for svo_position in range(50):
        # Retrieve left image from SVO
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            frame = left_image.get_data()

            # Prepare and normalize image for inference
            im = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            resized_im = cv2.resize(im, MODEL_INPUT_SIZE)
            normalized_im = normalize(resized_im)

            # Inference
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            inputs[0].host = np.ascontiguousarray(normalized_im)
            outputs = common_runtime.do_inference(
                context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
            )

            #Post processing
            h = MODEL_INPUT_SIZE[0]
            w = MODEL_INPUT_SIZE[1]

            outputs = outputs[0].reshape((1, 84, 8400))

            output = outputs[0].transpose()

            boxes, confidences, class_ids, depths = [], [], [], []
            for detection in output:
                confidence = np.max(detection[4:])
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                class_id = np.argmax(detection[4:])

                x_center, y_center, width, height = detection[:4]
                frame_height, frame_width = frame.shape[:2]
                x_min = int((x_center - width / 2) / w * frame_width)
                y_min = int((y_center - height / 2) / h * frame_height)
                x_max = int((x_center + width / 2) / w * frame_width)
                y_max = int((y_center + height / 2) / h * frame_height)

                x = int(x_center / w * frame_width)
                y = int(y_center / h * frame_height)

                zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
                depth_value = depth_map.get_value(x, y)[1]

                boxes.append([x_min, y_min, x_max, y_max])
                confidences.append(float(confidence))
                class_ids.append(yolo_classes[class_id])
                depths.append(depth_value)

            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            final_boxes = [boxes[i] for i in indices]
            final_confidences = [confidences[i] for i in indices]
            final_class_ids = [class_ids[i] for i in indices]
            final_depths = [depths[i] for i in indices]

            #Annotate
            for box, confidence, class_id, depth in zip(final_boxes, final_confidences, final_class_ids, final_depths):
                x_min, y_min, x_max, y_max = box
                label = f"{class_id:} CONFIDENCE: {confidence:.2f} DEPTH: {depth:.2f}"
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Detections", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        if zed.get_svo_position() >= nb_frames - 1:
            print("\nEnd of SVO file reached. Exiting.")
            break

    # Release resources
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()