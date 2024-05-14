import csv
import os
import cv2
import random

import pandas as pd
import numpy as np
import mediapipe as mp

import torch
from torch.utils.data import Dataset




# Dataset for landmarks processing
class HandLandmarksDataset(Dataset):

    def __init__(self, annotation, train = False, num_classes = 51):
        '''
        args:
            annotation: string(path to annotation with path to image and target),
            train: bool,
            num_classes: int
        return:
            normalized mediapipe landmarks,
            target
        '''
        self.holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=True)
        self.train = train
        self.annotation = pd.read_csv(annotation)
        self.num_classes = num_classes
        self.image_paths = self.annotation['attachment_id'].values
        self.labels = self.annotation['text'].values


    def __len__(self):
        return len(self.image_paths)


    def concatinate_features(self, left_hand_pos, right_hand_pos, body_pos):
        '''
        input:
            left_hand_pos: landmarks of left hand
            right_hand_pos: landmarks of right hand
            body_pos: landmarks of body
        retuern:
            concatinated_pos: (21+21+23, 2) shaped np.array, filled with fill_value if None
        '''

        num_hand_landmarks = 21
        num_body_landmarks = 23
        num_dimensions = 2
        fill_value=0.

        concatenated_pos = np.full(((num_hand_landmarks * 2) + num_body_landmarks, num_dimensions), fill_value)

        if left_hand_pos is not None:
            concatenated_pos[:num_hand_landmarks] = left_hand_pos

        if right_hand_pos is not None:
            concatenated_pos[num_hand_landmarks:(num_hand_landmarks*2)] = right_hand_pos

        concatenated_pos[(num_hand_landmarks*2):] = body_pos

        return concatenated_pos


    def pre_process_landmarks(self, concatenated_pos, body_pos):
        '''
        Normalize landmarks depending of body[11] (left shoulder landmark) and max_value
        input:
            concatenated_pos: output from concatinate_features
            body_pos: landmarks of body
        return:
            concatinated_1d: Normalised 1D array of landmarks
        '''
        concatinated_1d = []
        if body_pos is None:
            base_x, base_y = concatenated_pos[0]
        else:
            base_x, base_y = body_pos[11]
        for index, landmark_point in enumerate(concatenated_pos):
            if landmark_point[0] == 0 and landmark_point[1] == 0:
                concatinated_1d.append(0.0)
                concatinated_1d.append(0.0)
                continue

            concatinated_1d.append(concatenated_pos[index][0] - base_x + 0.1)
            concatinated_1d.append(concatenated_pos[index][1] - base_y + 0.1)

        # Normalization
        max_value = max(concatinated_1d)


        def normalize_(n):
            if max_value ==0:
                return 0
            return n / max_value

        concatinated_1d = list(map(normalize_, concatinated_1d))

        return torch.tensor(concatinated_1d, dtype=torch.float32)


    def extract_landmarks(self, results):
        left_hand_pos = results.left_hand_landmarks
        right_hand_pos = results.right_hand_landmarks
        body_pos = results.pose_landmarks
        num_body_landmarks = 23

        # Getting body keypoints:
        if body_pos is not None:
            body_pos = body_pos.landmark
            body_pos = np.array([[body_pos[indx].x, body_pos[indx].y] for indx in range(num_body_landmarks)])


        # Making landmarks in array of shape (21, 2) 
        if left_hand_pos is not None:
            left_hand_pos = np.array([[landmark.x, landmark.y] for landmark in left_hand_pos.landmark])
        if right_hand_pos is not None:
            right_hand_pos = np.array([[landmark.x, landmark.y] for landmark in right_hand_pos.landmark])


        return left_hand_pos, right_hand_pos, body_pos


    def random_flip(self, image):
        if np.random.choice([True, False]):
            return cv2.flip(image, 1)
        return image


    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.train:
            image = self.random_flip(image)
        if image is None:
            raise IOError(f"Failed to read image: {image_path}")

        landmarks = self.holistic.process(image)
        left_hand_features, right_hand_features, body_features = self.extract_landmarks(landmarks)
        features = self.concatinate_features(left_hand_features, right_hand_features, body_features)
        features = self.pre_process_landmarks(features, body_features)
        return torch.tensor(features, dtype=torch.float32), label

