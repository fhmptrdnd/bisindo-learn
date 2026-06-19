import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gtts import gTTS
import pygame
import os
import joblib
import numpy as np
import pyttsx3
import threading

print("Memuat model...")
clf = joblib.load('model_isyarat.pkl')

model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

kata_terketik = ""
prediksi_sebelumnya = ""
frame_count = 0
THRESHOLD = 20

# inisialisasi mixer audio dari pygame
pygame.mixer.init()

def ucapkan_teks(teks):
    if teks.strip() != "":
        try:
            tts = gTTS(text=teks, lang='id', slow=False)
            file_audio = "suara_temp.mp3"
            tts.save(file_audio)
            
            pygame.mixer.music.load(file_audio)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            pygame.mixer.music.unload()
            os.remove(file_audio)
            
        except Exception as e:
            print(f"[ERROR TTS] Pastikan komputer terhubung ke internet. Detail: {e}")

print("Kamera menyala! Tunjukkan isyarat tanganmu.")
print("=== KONTROL KEYBOARD ===")
print("[SPASI] = Spasi Kata | [BACKSPACE] = Hapus Huruf")
print("[ENTER] = Bicara (TTS) | [C] = Bersihkan Layar | [ESC] = Keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    huruf_terdeteksi = ""
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            koordinat = []
            for landmark in hand_landmarks:
                koordinat.extend([landmark.x, landmark.y, landmark.z])
                
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            

            input_data = np.array(koordinat).reshape(1, -1)
            prediksi = clf.predict(input_data)
            huruf_terdeteksi = prediksi[0]

    if huruf_terdeteksi:
        if huruf_terdeteksi == prediksi_sebelumnya:
            frame_count += 1
        else:
            frame_count = 0
            prediksi_sebelumnya = huruf_terdeteksi

        if frame_count == THRESHOLD:
            kata_terketik += huruf_terdeteksi
            frame_count = -15 

    else:
        frame_count = 0
        prediksi_sebelumnya = ""

    cv2.rectangle(frame, (20, 20), (120, 120), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, huruf_terdeteksi, (45, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 5)
    
    if frame_count > 0:
        progress_width = int((frame_count / THRESHOLD) * 100)
        cv2.rectangle(frame, (20, 120), (20 + progress_width, 130), (0, 255, 255), cv2.FILLED)

    cv2.rectangle(frame, (20, 400), (620, 460), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, kata_terketik, (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('Penerjemah Bahasa Isyarat', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # 'ESC'
        break
    elif key == 8: # 'backspace'
        kata_terketik = kata_terketik[:-1]
    elif key == 32: # 'spasi'
        kata_terketik += " "
    elif key == ord('c') or key == ord('C'): # 'C'
        kata_terketik = ""
    elif key == 13: # enter
        suara_thread = threading.Thread(target=ucapkan_teks, args=(kata_terketik,))
        suara_thread.start()

cap.release()
cv2.destroyAllWindows()