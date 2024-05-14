import csv
import os
import cv2
import random

import pandas as pd
import numpy as np
import mediapipe as mp

import torch
import torch.nn as nn


from dataset.builder import *
from model.buider import *

from sklearn.preprocessing import LabelEncoder
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")


decoder_classess = np.load('./data/encodeers/decode_final.npy', allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = decoder_classess

input_size = 65 * 2 
num_classes = 51
model = build_model(input_size, num_classes)
num_epochs = 20

model.load_state_dict(torch.load('./model/last_run/model_20.pth'))
model.to('cuda')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=True)

# Predicting landmarks and drawing 
def detect_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)
    
    
    
    
    # Right hand
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )

    # Left Hand
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )

    # Pose Detections
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
    
    


    return frame, results


# Normalization and preprocessing done in dataset
def transform(results):
    
    left_hand_pos = results.left_hand_landmarks
    right_hand_pos = results.right_hand_landmarks
    body_pos = results.pose_landmarks
    
    fill_value = 0.
    
    num_hand_landmarks = 21
    num_body_landmarks = 23
    num_dimensions = 2
    
    image_width, image_height = 512, 512
    landmark_list = []
    
    # Getting body keypoints:
    if body_pos is not None:
        body_pos = body_pos.landmark
        body_pos = np.array([[body_pos[indx].x, body_pos[indx].y] for indx in range(num_body_landmarks)])
    
    
    # Making landmarks in array of shape (21, 2) 
    if left_hand_pos is not None:
        left_hand_pos = np.array([[landmark.x, landmark.y] for landmark in left_hand_pos.landmark]) # ignore landmark z
    if right_hand_pos is not None:
        right_hand_pos = np.array([[landmark.x, landmark.y] for landmark in right_hand_pos.landmark]) # ignore landmark z
                  

    # Creating with filled "fill_value" array for placing landmarks in
    concatenated_pos = np.full(((num_hand_landmarks * 2) + num_body_landmarks, num_dimensions), fill_value)
    
    if left_hand_pos is not None:
        concatenated_pos[:num_hand_landmarks] = left_hand_pos

    if right_hand_pos is not None:
        concatenated_pos[num_hand_landmarks:(num_hand_landmarks*2)] = right_hand_pos
        
    concatenated_pos[(num_hand_landmarks*2):] = body_pos
    
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



# Real Time excecution
cap = cv2.VideoCapture(0)
previous = ' '
    
while cap.isOpened():
    ret, frame = cap.read()

    image = cv2.resize(frame, (512, 512))
    # Make Detections
    image, result = detect_landmarks(image)

    input_data = transform(result).view(-1, 65*2).cuda() 
    with torch.no_grad():
        output = model(input_data)

    if torch.max(output) > 0.8:
        
        prediction = output.argmax(dim=1).item()
        if previous != prediction:
            previous = prediction
        out_text = label_encoder.inverse_transform([previous])[0]
    else:
        out_text = ' '
    cv2.putText(image, out_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.imshow('Raw Webcam Feed', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
