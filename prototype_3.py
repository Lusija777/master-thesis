from scipy.optimize import linear_sum_assignment
import numpy as np
import os
import json

# PRObLEM
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class SpaceDataset():
    def __init__(self, root_folder):
        self.series_data = []

        for subfolder in os.listdir(root_folder):
            subfolder_path = os.path.join(root_folder, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            input_folder = os.path.join(subfolder_path, 'input')
            output_folder = os.path.join(subfolder_path, 'output')

            photos_one_serie = []  # List of frames (each frame is a list of objects)

            if os.path.isdir(input_folder):
                for json_file in os.listdir(input_folder):
                    json_path = os.path.join(input_folder, json_file)
                    if json_file.endswith('.json'):
                        time = None
                        json_content = load_json(json_path)
                        one_photo = []  # Store object positions in this frame

                        for object in json_content.get("frame_objects", []):
                            if object.get("star") == True:
                                continue  # Ignore stars
                            time = object.get("mjd")
                            ecs_cords = object.get("ecs_coords", {})
                            one_photo.append([
                                ecs_cords.get("ra"),
                                ecs_cords.get("dec")
                            ])

                        photos_one_serie.append(one_photo)

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

            self.series_data.append((photos_one_serie, output_tracklet))

# Define distance function between two points (RA, Dec)
def distance(point1, point2):
    ra1, dec1 = point1
    ra2, dec2 = point2
    return np.sqrt((ra1 - ra2) ** 2 + (dec1 - dec2) ** 2)

root_directory = "series_one_object"
dataset = SpaceDataset(root_directory)
serie = dataset.series_data[0]
# Example positions of detected objects in two consecutive frames
frame_1_positions = serie[0][0]  # Frame 1
frame_2_positions = serie[0][1]

# Create cost matrix: calculate the distance between each pair of objects across frames
cost_matrix = np.array([[distance(p1, p2) for p2 in frame_2_positions] for p1 in frame_1_positions])

# Solve the assignment problem (minimizing cost)
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# row_ind and col_ind give the best association of objects across frames
for r, c in zip(row_ind, col_ind):
    print(f"Object in Frame 1 at {frame_1_positions[r]} is matched with object in Frame 2 at {frame_2_positions[c]}")
