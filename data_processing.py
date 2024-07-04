import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pickle
import numpy as np
DATA_DIR = 'data'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

data = []
labels = []
for label in os.listdir(DATA_DIR):
    for image_path in os.listdir(os.path.join(DATA_DIR, label)):
        data_aux = []
        image = cv2.imread(os.path.join(DATA_DIR, label, image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                
            
            data.append(data_aux)
            labels.append(int(label))

f = open("data.pickle", 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
print("Labels: ", np.unique(labels))  
f.close()