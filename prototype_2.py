import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# === 1. Funkcia na načítanie dát ===

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
                                ecs_cords.get("dec"),
                                time
                            ])

                        input_series.append(frame_data)

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

            self.series_data.append((np.array(input_series, dtype=np.float32), np.array(output_tracklet, dtype=np.float32)))


def load_data(json_files):
    all_objects = []
    timestamps = []

    for t, file in enumerate(json_files):
        with open(file, 'r') as f:
            data = json.load(f)

        for obj in data["objects"]:
            ra, dec = obj["ra"], obj["dec"]
            all_objects.append([ra, dec])
            timestamps.append(t)

    return np.array(all_objects), np.array(timestamps)

# === 2. Načítanie dát ===
json_files = ["data1.json", "data2.json"]
X, timestamps = load_data(json_files)

# Normalizácia dát
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Definícia Autoencoderu na extrakciu príznakov ===
encoding_dim = 2  # Redukujeme na 2D priestor

input_layer = keras.Input(shape=(2,))
encoded = layers.Dense(4, activation="relu")(input_layer)
encoded = layers.Dense(encoding_dim, activation="relu")(encoded)

decoded = layers.Dense(4, activation="relu")(encoded)
decoded = layers.Dense(2, activation="linear")(decoded)

autoencoder = keras.Model(input_layer, decoded)
encoder = keras.Model(input_layer, encoded)  # Extrahujeme len kódovaciu časť

autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=16, shuffle=True)

# === 4. Extrakcia príznakov a klastrovanie DBSCAN ===
features = encoder.predict(X_scaled)
clustering = DBSCAN(eps=0.5, min_samples=2).fit(features)

# === 5. Zoskupenie objektov do trajektórií ===
trajectories = {}
for obj_idx, cluster_id in enumerate(clustering.labels_):
    if cluster_id == -1:
        continue  # -1 znamená "šum" = nebol nájdený ako súčasť žiadneho klastru

    if cluster_id not in trajectories:
        trajectories[cluster_id] = []

    trajectories[cluster_id].append((timestamps[obj_idx], X[obj_idx]))

# === 6. Výpis trajektórií ===
for cluster_id, trajectory in trajectories.items():
    trajectory.sort()  # Usporiadame podľa času
    print(f"Objekt {cluster_id}: {trajectory}")
