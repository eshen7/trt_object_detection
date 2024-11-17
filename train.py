from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

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

def normalize_data(dat):
    scaler = MinMaxScaler()
    original_shape = dat.shape
    dat = dat.reshape(-1, 2)
    scaler.fit(dat)
    dat = scaler.transform(dat)
    return dat.reshape(original_shape), scaler

# Generate sample data
def generate_data(sequence_length=10):
    with open('trajectory_data.json', 'r') as f:
        data = json.load(f)
    track_sequences = defaultdict(list)

    # Find the maximum frame to know how long each sequence should be
    max_frame = max(item['frame'] for item in data)

    # Process each frame
    for frame_data in data:
        frame = frame_data['frame']
        detections = frame_data['detections']

        # Collect centers for each track_id
        centers = {detection['track_id']: detection['center'] for detection in detections}

        for track_id in centers.keys():
            track_sequences[track_id].append(centers[track_id])

    sequences = []

    for track_id in track_sequences:
        if len(track_sequences[track_id]) > sequence_length:
            for i in range(len(track_sequences[track_id]) - sequence_length):
                sequences.append(track_sequences[track_id][i:i + sequence_length])

    return np.array(sequences)


# Set parameters
input_size = 2
hidden_size = 128
output_size = 2
sequence_length = 10
num_samples = 1000
num_epochs = 100
batch_size = 16

# Prepare data
data = generate_data()
data, scaler = normalize_data(data)
X = torch.FloatTensor(data[:, :-3, :])  # Use sequence except last for input
y = torch.FloatTensor(data[:, 3:, :])   # Use sequence except first for target
# DataLoader for batching
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create and train the model
model = TrajectoryLSTM(input_size, hidden_size, output_size, scaler)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}')

model.eval()

torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler  # Save the scaler for normalization/denormalization
}, "trajectory_model.pth")