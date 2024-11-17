import argparse
import cv2
import numpy as np
import tensorrt as trt
import pyzed.sl as sl
import sys
import common
import common_runtime
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from deep_sort_realtime.deepsort_tracker import DeepSort

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

tracking_history = {}

MAX_CENTERS = 7
FRAME_TIMEOUT = 5

CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.7
MODEL_INPUT_SIZE = (640, 640)

# Initialize Deep SORT
tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=NMS_THRESHOLD)

# Define the LSTM model
class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, scaler):
        super(TrajectoryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.scaler = scaler

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)  # Predict all timesteps
        return output

    def predict(self, x):
        x = x[0]
        x = self.scaler.transform(x)
        x = torch.from_numpy(x).unsqueeze(0).float()
        output = self.forward(x)[0][-1]
        output_reshaped = output.view(-1, 2).cpu().detach().numpy()
        denormalized_output = self.scaler.inverse_transform(output_reshaped)
        return denormalized_output.reshape(output.shape)

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
    parser.add_argument("--traj_pred_model", type=str, required=True, help='Path to the .pt file')
    parser.add_argument('--resolution', type=str, help='Resolution: HD2K, HD1200, HD1080, HD720, SVGA, VGA',
                        default='HD720')
    return parser.parse_args()

def setup_zed_camera(opt):
    init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA,
                             coordinate_units=sl.UNIT.MILLIMETER,
                             coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)

    # Set input SVO file and resolution
    init.set_from_svo_file(opt.input_svo_file)
    resolution_dict = {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1200": sl.RESOLUTION.HD1200,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "SVGA": sl.RESOLUTION.SVGA,
        "VGA": sl.RESOLUTION.VGA
    }
    init.camera_resolution = resolution_dict.get(opt.resolution.upper(), sl.RESOLUTION.HD720)
    zed = sl.Camera()
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("Error opening ZED Camera")
        zed.close()
        sys.exit(1)
    return zed

def main():
    args = parse_args()

    yolo_classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    # Load TensorRT model
    engine = get_engine(args.model)
    context = engine.create_execution_context()

    #output file details
    output_file = 'output_video.avi'
    frame_width = 1920
    frame_height = 1200
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'mp4v' for .mp4
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Initialize ZED camera from SVO
    zed = setup_zed_camera(args)
    runtime_parameters = sl.RuntimeParameters()
    depth_map = sl.Mat()
    left_image = sl.Mat()

    nb_frames = zed.get_svo_number_of_frames()

    model_state_dict = torch.load(args.traj_pred_model)
    scaler = model_state_dict['scaler']
    model = TrajectoryLSTM(2, 128, 2, scaler)
    model.load_state_dict(model_state_dict['model_state_dict'])
    model.eval()

    trajectory_data = []

    for svo_position in range(250):
        frame_data = {"frame": svo_position, "detections": []}
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            frame = left_image.get_data()
            im = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            resized_im = cv2.resize(im, MODEL_INPUT_SIZE)
            normalized_im = normalize(resized_im)

            # Inference
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            inputs[0].host = np.ascontiguousarray(normalized_im)
            outputs = common_runtime.do_inference(
                context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
            )

            h, w = MODEL_INPUT_SIZE
            outputs = outputs[0].reshape((1, 84, 8400))[0].transpose()

            boxes, confidences, class_ids, depths = [], [], [], []
            for detection in outputs:
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

                boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                depths.append(depth_value)

            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            detections = [[boxes[i], confidences[i], class_ids[i]] for i in indices]
            depths = [depths[i] for i in indices]

            # Update DeepSORT tracker
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            tracks = tracker.update_tracks(detections, frame=frame_rgb, others=depths)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                bbox = track.to_ltwh()
                depth = track.get_det_supplementary()
                x_min, y_min, width, height = bbox.astype("int")
                center = [int(x_min + width / 2), int(y_min + height / 2)]

                # Update tracking history
                if track_id not in tracking_history:
                    tracking_history[track_id] = {
                        "centers": [],
                        "last_seen": svo_position
                    }
                tracking_history[track_id]["centers"].append(center)
                tracking_history[track_id]["last_seen"] = svo_position
                if len(tracking_history[track_id]["centers"]) > MAX_CENTERS:
                    tracking_history[track_id]["centers"].pop(0)
                cv2.rectangle(frame_rgb, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 2)
                cv2.putText(frame_rgb, f"ID {track_id} Depth: {depth}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                track_data = {
                    "track_id": int(track_id),
                    "center": [int(x_min + width / 2), int(y_min + height / 2)]
                }
                frame_data["detections"].append(track_data)

            stale_tracks = [tid for tid, data in tracking_history.items() if
                            svo_position - data["last_seen"] > FRAME_TIMEOUT]
            for stale_track in stale_tracks:
                del tracking_history[stale_track]

            can_be_predicted = [data for tid, data in tracking_history.items() if len(data["centers"]) == MAX_CENTERS]

            for data in can_be_predicted:
                centers = data["centers"]
                data = torch.from_numpy(np.array(centers)).unsqueeze(0).float()
                predictions = []
                for _ in range(10):
                    pred = model.predict(data)
                    predictions.append(pred)
                    data = torch.cat((data[:, 1:, :], torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)), dim=1)
                predictions = np.array(predictions).squeeze()
                for prediction in predictions:
                    cv2.circle(frame_rgb, (int(prediction[0]), int(prediction[1])), 1, (255, 0, 0), 2)

            trajectory_data.append(frame_data)

            # Display
            cv2.imshow("Detections", frame_rgb)
            out.write(frame_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if zed.get_svo_position() >= nb_frames - 1:
            break

    with open("trajectory_data.json", "w") as f:
        json.dump(trajectory_data, f, indent=4)

    # Release resources
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
