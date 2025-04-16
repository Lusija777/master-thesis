import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence



def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class SpaceDataset(Dataset):
    def __init__(self, root_folder):
        self.series_data = []

        for subfolder in os.listdir(root_folder):
            subfolder_path = os.path.join(root_folder, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            input_folder = os.path.join(subfolder_path, 'input')
            output_folder = os.path.join(subfolder_path, 'output')

            input_series = []  # List of frames (each frame is a list of objects)

            if os.path.isdir(input_folder):
                for json_file in os.listdir(input_folder):
                    json_path = os.path.join(input_folder, json_file)
                    if json_file.endswith('.json'):
                        time = None
                        json_content = load_json(json_path)
                        frame_data = []  # Store object positions in this frame

                        for object in json_content.get("frame_objects", []):
                            if object.get("star") == True:
                                continue  # Ignore stars
                            time = object.get("mjd")
                            ecs_cords = object.get("ecs_coords", {})
                            frame_data.append([
                                ecs_cords.get("ra"),
                                ecs_cords.get("dec")
                            ])

                        input_series.append((time,frame_data))

            num_photos = len(input_series)

            output_tracklet = []
            if os.path.isdir(output_folder):
                for json_file in os.listdir(output_folder):
                    json_path = os.path.join(output_folder, json_file)
                    if json_file.endswith('.json'):
                        json_content = load_json(json_path)
                        for i, object in enumerate(json_content.get("frame_objects", [])):
                            ecs_cords = object.get("ecs_coords", {})
                            output_tracklet.append([
                                ecs_cords.get("ra"),
                                ecs_cords.get("dec"),
                                i
                            ])
                        break  # Only one file is expected

            input_series.sort(key=lambda x: x[0])
            input_data = []

            for i, photo in enumerate(input_series):
                time, frame_data = photo
                for object in frame_data:
                    object.append(i/num_photos)
                    input_data.append(object)

            self.series_data.append((np.array(input_data, dtype=np.float32), np.array(output_tracklet, dtype=np.float32), num_photos))


    def __len__(self):
        return len(self.series_data)

    def __getitem__(self, idx):
        input_seq, output_seq, num_photos = self.series_data[idx]
        return input_seq, output_seq, num_photos


class TrackletPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TrackletPredictionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Output is 3: RA, Dec, and time

    def forward(self, x, num_photos):
        batch_size, total_objects, _ = x.shape  # `total_objects = num_photos * num_objects`

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)

        # Apply FC layer to transform hidden_size → 3 (RA, Dec, Time)
        out = self.fc(lstm_out)  # Shape: (batch_size, num_photos * num_objects, 3)

        # Extract the photo_id column
        photo_ids = x[:, :, 2].long()  # Shape: (batch_size, num_photos * num_objects)

        photo_outputs = torch.zeros(batch_size, num_photos, 3, device=x.device)
        photo_counts = torch.zeros(batch_size, num_photos, 1, device=x.device)

        # Accumulate predictions per photo
        for b in range(batch_size):
            for i in range(total_objects):  # Instead of `num_photos * num_objects`
                photo_id = photo_ids[b, i]  # Get the photo index
                if photo_id < num_photos:  # Avoid indexing errors
                    photo_outputs[b, photo_id] += out[b, i]  # Sum predictions for this photo
                    photo_counts[b, photo_id] += 1  # Count objects per photo

        # Avoid division by zero
        photo_counts[photo_counts == 0] = 1

        # Compute the mean prediction for each photo
        final_output = photo_outputs / photo_counts  # Shape: (batch_size, num_photos, 3)

        return final_output



def custom_collate_fn(batch):
    # Batch is a list of (input_data, output_data) tuples
    inputs, outputs, photo_number = zip(*batch)

    input_tensors = [torch.tensor(input_data, dtype=torch.float32) for input_data in inputs]
    input_tensor = pad_sequence(input_tensors, batch_first=True, padding_value=0.0)  # Pad with zeros

    output_tensors = [torch.tensor(output_data, dtype=torch.float32) for output_data in outputs]
    output_tensor = pad_sequence(output_tensors, batch_first=True, padding_value=0.0)  # Pad with zeros

    return input_tensor, output_tensor, photo_number[0]

if __name__ == "__main__":
    root_directory = "series_one_object"
    dataset = SpaceDataset(root_directory)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    model = TrackletPredictionModel(input_size=3, hidden_size=64,
                                    output_size=3)  # 3 for RA, Dec, and time
    criterion = nn.MSELoss()  # Assuming we're using MSE for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            x, y, length = batch  # Now we're unpacking the lengths tensor

            optimizer.zero_grad()

            # Forward pass
            predictions = model(x, length)

            # Compute loss
            loss = criterion(predictions, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
