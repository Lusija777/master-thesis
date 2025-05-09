import os
import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


# === 1. Funkcia na načítanie dát ===
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# === 2. Načítanie dát do štruktúry ===
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
                                ecs_cords.get("dec"),
                                time
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


# === 3. Algoritmus na sledovanie objektov ===
def track_objects_across_frames(serie):
    object_tracks = defaultdict(list)

    # Pre každú snímku v sérii, iterujeme cez objekty
    for frame_idx, frame in enumerate(serie):
        for obj in frame:
            ra, dec, mjd = obj
            # Môžeme pridať objekt do listu sledovaných objektov
            object_tracks[(ra, dec)].append({
                'frame_idx': frame_idx,
                'mjd': mjd
            })

    return object_tracks


# === 4. Klastrovanie objektov na základe RA a DEC ===
def cluster_objects(ra_dec_data):
    # Používame DBSCAN pre klastrovanie objektov podľa RA a DEC
    X = np.array(ra_dec_data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(eps=0.1, min_samples=2)
    labels = db.fit_predict(X_scaled)

    return labels


# === 5. Sledovanie trajektórií podľa klastrov ===
def group_trajectories_by_cluster(object_tracks, ra_dec_data, labels):
    grouped_trajectories = defaultdict(list)

    for idx, label in enumerate(labels):
        if label != -1:  # Ignorujeme šum
            ra, dec = ra_dec_data[idx]
            grouped_trajectories[label].append({
                'ra': ra,
                'dec': dec,
                'track_data': object_tracks[(ra, dec)]
            })

    return grouped_trajectories


# === 6. Hlavný kód ===
root_directory = "series_one_object"
dataset = SpaceDataset(root_directory)
serie = dataset.series_data[0][0]

# Sledovanie objektov cez všetky snímky
object_tracks = track_objects_across_frames(serie)

# Získanie dát pre RA, DEC
ra_dec_data = []

for frame in serie:
    for obj in frame:
        ra, dec, _ = obj
        ra_dec_data.append([ra, dec])

# Klastrovanie objektov podľa RA a DEC
labels = cluster_objects(ra_dec_data)

# Zoskupovanie trajektórií podľa klastrov
grouped_trajectories = group_trajectories_by_cluster(object_tracks, ra_dec_data, labels)

# Výpis výsledkov
for cluster_id, trajectories in grouped_trajectories.items():
    print(f"Cluster {cluster_id}:")
    for trajectory in trajectories:
        print(f"  RA: {trajectory['ra']:.6f}, DEC: {trajectory['dec']:.6f}")
        for track in trajectory['track_data']:
            print(f"    Frame {track['frame_idx']}, Time (MJD): {track['mjd']:.6f}")
