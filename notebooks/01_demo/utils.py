import pandas as pd
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from tqdm import tqdm
import subprocess
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms



def load_examples(folder_path):
    mp4_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mp4"):
                mp4_files.append(os.path.join(root, file))
    return mp4_files


def extract_frames(video_path):
    cap = cv.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)


def resample_frames(frames, num_output_frames=20):
    total_frames = len(frames)
    indices = np.linspace(0, total_frames - 1, num=num_output_frames, dtype=int)
    resampled_frames = frames[indices]
    return resampled_frames


def write_video(output_path, frames, fps=20):
    height, width, layers = frames[0].shape
    size = (width, height)
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, size, isColor=True)
    
    for frame in frames:
        out.write(frame)

    out.release()
    
    
def estimate_poses(filepaths, model, json_output_dir):
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    for fp in tqdm(filepaths, desc="Processing videos"):

        video_results_gen = model.track(fp, stream=True, verbose=False)

        track_by_id = defaultdict(lambda: [])
        track_kps_by_id = defaultdict(lambda: [])
        track_boxes_by_id = defaultdict(lambda: [])
        frame_id = 0
        for frame_results in video_results_gen:
            if frame_results.boxes.id is not None:
                keypoints = frame_results.keypoints.cpu()
                boxes_xywh = frame_results.boxes.xywh.cpu()
                boxes_full = frame_results.boxes.cpu()
                track_ids = frame_results.boxes.id.int().cpu().tolist()
                for kps, bxs, bxs_xywh, track_id in zip(keypoints, boxes_full, boxes_xywh, track_ids):
                    track_kps_by_id[track_id].append(kps)
                    track_boxes_by_id[track_id].append(bxs_xywh)
                    # print(bxs)
                    # print(kps)
                    track_by_id[track_id].append({
                        'video_path': fp,
                        'frame_id': frame_id,
                        'img_shape': bxs.orig_shape,
                        'class': bxs.cls.int().item(),
                        'class_conf': bxs.conf.item(),
                        'boxes_xywh': bxs.xywh.tolist()[0],
                        'boxes_xywhn': bxs.xywhn.tolist()[0],
                        'boxes_xyxy': bxs.xyxy.tolist()[0],
                        'boxes_xyxyn': bxs.xyxyn.tolist()[0],
                        'keypoints_xy': kps.xy.tolist()[0],
                        'keypoints_xyn': kps.xyn.tolist()[0],
                        'keypoints_conf': kps.conf.tolist()[0],
                    })

            frame_id += 1

        if not os.path.exists(json_output_dir):
            os.makedirs(json_output_dir)
            
        pretty_data = json.dumps(track_by_id, indent=4)
        with open(f"{os.path.join(json_output_dir, os.path.basename(fp[:-4]))}.json", 'w') as json_file:
            json.dump(track_by_id, json_file, indent=4)
            
            
def draw_bounding_boxes_on_video(video_path, output_path, pose_data_json_path, track_ids=None, draw_bb=True, draw_pose=False):
    with open(pose_data_json_path, 'r') as file:
        pose_data = json.load(file)

    cap = cv.VideoCapture(video_path)
    
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, frame_id, pose_data, track_ids, draw_bb, draw_pose)

        out.write(frame)

        frame_id += 1
    
    cap.release()
    out.release()
    

KEYPOINT_CONNECTIONS = [
    (0, 1), (1, 3),  # Nose to Left Eye to Left Ear
    (0, 2), (2, 4),  # Nose to Right Eye to Right Ear
    (5, 6),  # Left Shoulder to Right Shoulder
    (5, 7), (7, 9),  # Left Shoulder to Left Elbow to Left Wrist
    (6, 8), (8, 10),  # Right Shoulder to Right Elbow to Right Wrist
    (5, 11), (6, 12),  # Left Shoulder to Left Hip and Right Shoulder to Right Hip
    (11, 12),  # Left Hip to Right Hip
    (11, 13), (13, 15),  # Left Hip to Left Knee to Left Ankle
    (12, 14), (14, 16)  # Right Hip to Right Knee to Right Ankle
]

KEYPOINT_COLORS = [
    # 0: Nose    1: L Eye      2: R Eye       3: L Ear       4: R Ear
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), 
    # 5: L Shoulder 6: R Shoulder 7: L Elbow      8: R Elbow      9: L Wrist
    (255, 0, 255), (128, 128, 0), (128, 0, 128), (0, 128, 128), (255, 128, 0),
    # 10: R Wrist   11: L Hip       12: R Hip     13: L Knee     14: R Knee
    (0, 255, 128), (128, 0, 255), (255, 0, 128), (128, 255, 0), (0, 128, 255),
    # 15: L Ankle     16: R Ankle
    (128, 128, 128), (255, 255, 255)
]

CONNECTION_COLORS = [
    (0, 255, 255), (255, 0, 255),  # (0, 1), (1, 3)
    (255, 255, 0), (128, 128, 255),  # (0, 2), (2, 4)
    (255, 128, 128),  # (5, 6)
    (128, 255, 128), (128, 128, 128),  # (5, 7), (7, 9)
    (128, 255, 128), (128, 128, 128),  # (6, 8), (8, 10)
    (255, 128, 255), (128, 255, 255),  # (5, 11), (6, 12)
    (255, 255, 128), (192, 192, 192),  # (11, 12)
    (64, 64, 64), (0, 0, 128),  # (11, 13), (13, 15)
    (128, 0, 0), (0, 128, 64)  # (12, 14), (14, 16)
]

def process_frame(frame, frame_id, pose_data, track_ids=None, draw_bb=True, draw_pose=True):
    for track_id, track_info in pose_data.items():
        if track_ids == None or track_id in track_ids:
            for frame_info in track_info:
                if frame_info['frame_id'] == frame_id:
                    if draw_bb:
                        x0, y0, x1, y1 = frame_info['boxes_xyxy']
                        # Draw the bounding box and track ID
                        cv.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
                        cv.putText(frame, f'ID: {track_id}', (int(x0), int(y0) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if draw_pose:
                        # Get data for pose keypoints and connections
                        keypoints = frame_info['keypoints_xy']
                        valid_keypoints = {idx: (int(kp[0]), int(kp[1])) for idx, kp in enumerate(keypoints) if kp != [0, 0]}

                        # Draw pose connections
                        for i, (start_idx, end_idx) in enumerate(KEYPOINT_CONNECTIONS):
                            if start_idx in valid_keypoints and end_idx in valid_keypoints:
                                start_point = valid_keypoints[start_idx]
                                end_point = valid_keypoints[end_idx]
                                cv.line(frame, start_point, end_point, CONNECTION_COLORS[i], 2)

                        # Draw pose keypoints
                        for idx, point in valid_keypoints.items():
                            if idx > 5:
                                color = KEYPOINT_COLORS[idx]
                                cv.circle(frame, point, 5, color, -1)

    return frame


KPS = {
    "Nose": 0,
    "Left Eye": 1,
    "Right Eye": 2,
    "Left Ear": 3,
    "Right Ear": 4,
    "Left Shoulder": 5,
    "Right Shoulder": 6,
    "Left Elbow": 7,
    "Right Elbow": 8,
    "Left Wrist": 9,
    "Right Wrist": 10,
    "Left Hip": 11,
    "Right Hip": 12,
    "Left Knee": 13,
    "Right Knee": 14,
    "Left Ankle": 15,
    "Right Ankle": 16
}

def find_bottom_player_track_id(json_data):
    """
    Note: keypoints coordinates are upside-down if plotted. y is 0 on the top and frame height at the bottom.
    """
    
    # bottom_player_track_id = None

    pose_score_by_id = defaultdict(lambda: 0)
    for track_id, pose_data_list in json_data.items():
        # print(track_id)
        
        for pose_data in pose_data_list:
            
            fh, fw = pose_data['img_shape']
            # print(fw, fh)
            pose_data_xy = np.array(pose_data['keypoints_xy'])
            # print(pose_data_xy.shape)
            # print(pose_data_xy)
            
            # Check if player's legs are below 50% of the screen
            # print(pose_data_xy[:, 1]])
            if np.max(pose_data_xy[:, 1]) > (fh / 2):
                pose_score_by_id[track_id] += 1000
            else:
                pose_score_by_id[track_id] -= 2000
                
            # Additionally, add that max y to distinct between bottom poses
            pose_score_by_id[track_id] += np.max(pose_data_xy[:, 1]) / 2
            
            # Find the most consistent pose - add len(pose_data)
            # means adding number of frames pose is detected in
            pose_score_by_id[track_id] += len(pose_data) * 10
            # print("len(pose_data) * 10: ", len(pose_data) * 10)
            
            # Use pose height
            pose_score_by_id[track_id] += pose_data['boxes_xywh'][3] * 5
            # print("pose_data['boxes_xywh'][3] * 0.1", pose_data['boxes_xywh'][3] * 0.1)
            
            # Use class confidence
            pose_score_by_id[track_id] += pose_data['class_conf'] * 200
            # print("pose_data['class_conf'] * 100", pose_data['class_conf'] * 100)
            
            # Use bad rows (0, 0)
            # pose_score_by_id[track_id] += sum([1 for xy in pose_data['keypoints_xy'][5:]
            #                                    if xy == [0, 0]]) * 100
            
    # print(pose_score_by_id)
            
    return str(max(pose_score_by_id, key=pose_score_by_id.get))


def make_translation_image(skeleton):
    
    # based on https://arxiv.org/pdf/1704.05645.pdf
    # from 2D data

    c_0 = min([elem[0] for elem in skeleton ] + [0])  # TODO fix zero hack
    c_1 = min([elem[1] for elem in skeleton ] + [0])
    c_2 = min([elem[2] for elem in skeleton ] + [0])

    C_0 = max([elem[0] for elem in skeleton ] + [0])
    C_1 = max([elem[1] for elem in skeleton ] + [0])
    C_2 = max([elem[2] for elem in skeleton ] + [0])

    tmp = max([C_0 - c_0, C_1 - c_1, C_2 - c_2])

    r_column = []
    g_column = []
    b_column = []

    for joint in skeleton:

        if joint is None:
            r_column.append(0)
            g_column.append(0)
            b_column.append(0)

        else:
            p_r = int(np.floor(255 * (joint[0] - c_0) / tmp))
            r_column.append(p_r)

            p_g = int(np.floor(255 * (joint[1] - c_1) / tmp))
            g_column.append(p_g)

            p_b = int(np.floor(255 * (joint[2] - c_2) / tmp))
            b_column.append(p_b)

    return r_column, g_column, r_column


def load_pickle_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model


def load_cnn_model(cnn_model_class, state_dict_path):
    model = cnn_model_class()
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    return model


def run_sklearn_model(model, X_test):
    predictions = model.predict(X_test)
    return predictions


def run_cnn_model(model, cnn_img):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_img = cnn_img.to(device)
    with torch.no_grad():
        cnn_prediction = model(cnn_img)
    return cnn_prediction


def prepare_img_for_basic_model(action_image):
    action_image = action_image.astype(np.float32)
    basic_img = action_image.flatten().reshape(1, -1)
    return basic_img


def prepare_img_for_cnn(action_image):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.8143442273139954, 0.7186146974563599, 0.8145506978034973],
                             std=[0.18269290030002594, 0.18128100037574768, 0.182467520236969])
    ])

    cnn_img = image_transform(action_image.astype(np.float32) / 255.0)
    cnn_img = cnn_img.unsqueeze(0)
    return cnn_img


def get_bottom_player_pose_data(fp, json_folder="temp/json"):
    json_path = os.path.join(json_folder, os.path.basename(fp).replace('.mp4', '.json'))

    with open(json_path, 'r') as file:
        json_data = json.load(file)

    bottom_player_track_id = find_bottom_player_track_id(json_data)
    bottom_player_pose_data = json_data[bottom_player_track_id]
    return bottom_player_pose_data


def create_action_image(bottom_player_pose_data):
    r_channel = []
    g_channel = []
    b_channel = []

    for bppd in bottom_player_pose_data:
        a = np.array(bppd["keypoints_xy"][5:])
        b = np.array(bppd["keypoints_conf"][5:]).reshape(12, 1)
        c = np.hstack((a, b))
        r_column, g_column, b_column = make_translation_image(c)

        r_channel.append(r_column)
        g_channel.append(g_column)
        b_channel.append(b_column)

    action_image = cv.merge((np.asarray(r_channel), np.asarray(g_channel), np.asarray(b_channel)))
    return action_image