import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
import pygame
import time

pygame.mixer.init()
def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load("assets/alarm.mp3")
        pygame.mixer.music.play(-1)

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

model = tf.keras.models.load_model("drowsiness_model.keras")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def preprocess_image(file_path):
    gray = cv2.cvtColor(file_path, cv2.COLOR_BGR2GRAY)
    clah_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clah_img.apply(gray)
    resized_img = cv2.resize(enhanced_img, (64, 64))
    normalized_img = resized_img / 255.0
    final_input = np.expand_dims(normalized_img, axis=(0, -1))
    return final_input

cap = cv2.VideoCapture(0)
start_sleep_time = None
long_sleep = 0

print("kamera aktif, q untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    status_text = "mata tidak terdeteksi"
    text_persentase = ""
    color = (255,255,255)
    eyes_detected = False

    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray_face = gray_frame[y:y+h, x:x+w]
        roi_color_face = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray_face)

        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color_face, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
            roi_eye = roi_color_face[ey:ey+eh, ex:ex+ew]
            input_data = preprocess_image(roi_eye)
            prediction = model.predict(input_data, verbose=0)[0][0]
            eyes_detected = True

            persen_awake = prediction * 100
            persen_sleepy = (1 - prediction) * 100

            if prediction < 0.5:
                status_text = "Sleepy"
                color = (0,0,255)
                text_persentase = f"Sleepy: {persen_sleepy:.1f}%"
                if start_sleep_time is None:
                    start_sleep_time = time.time()
                else:
                    long_sleep = time.time() - start_sleep_time
            else:
                status_text = "Awake"
                color = (0,255,0)
                start_sleep_time = None
                long_sleep = 0
                text_persentase = f"Awake: {persen_awake:.1f}%"
            break
    
    if not eyes_detected and len(faces) > 0:
        status_text = "Sleepy (Mata Tertutup)"
        color = (0, 0, 255)
        text_persentase = "Sleepy: 100%"
        if start_sleep_time is None:
            start_sleep_time = time.time()
        else:
            long_sleep = time.time() - start_sleep_time
    elif len(faces) == 0:
        start_sleep_time = None
        long_sleep = 0

    if long_sleep > 2.0:
        cv2.putText(frame, "ALERT!", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        play_alarm()
    else:
        stop_alarm()

    cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, text_persentase, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_alarm()
cap.release()
cv2.destroyAllWindows()
