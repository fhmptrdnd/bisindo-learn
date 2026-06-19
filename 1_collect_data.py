import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os
import urllib.request

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Mendownload model AI MediaPipe terbaru (hand_landmarker.task)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download selesai!\n")

# tasks API mediaPipe
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# dataset
csv_file = 'dataset_isyarat.csv'
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

cap = cv2.VideoCapture(0)
print("Tekan tombol huruf (A-Z) untuk merekam data. Tekan 'ESC' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1) # mirror
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            koordinat = []
            
            # drawing utils buat sendi tangan
            for landmark in hand_landmarks:
                # titik x dan y punya Tasks API bentuknya rasio 0.0 - 1.0, dikali resolusi frame
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                
                # simpan data mentah
                koordinat.extend([landmark.x, landmark.y, landmark.z])
                
            key = cv2.waitKey(1) & 0xFF
            if ord('a') <= key <= ord('z'):
                huruf = chr(key).upper()
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    koordinat.insert(0, huruf)
                    writer.writerow(koordinat)
                print(f"Data {huruf} tersimpan!")

    cv2.imshow('Koleksi Data (Tasks API)', frame)
    if cv2.waitKey(1) & 0xFF == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()