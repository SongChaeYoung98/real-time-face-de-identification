'''
__author__ = 'Song Chae Young'
__date__ = 'Jan.18, 2023'
__email__ = '0.0yeriel@gmail.com'
__fileName__ = 'real-time-de-identification.py'
__github__ = 'SongChaeYoung98'
__status__ = 'Development'
'''

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.ndimage import gaussian_filter

# Color Code
RED = (0, 0, 255)
GREEN = (0, 255, 0)

# Instead of Streaming
VIDEO_FILE = 'datasets/etc/FaceVideo2.mp4'

# Detect Face
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load Model
model_name = 'face_classifier_ResNet152.h5'
face_classifier = keras.models.load_model(f'models/{model_name}')
class_names = ['me', 'not_me']


# Check the Coordinates will be Non-negative
def get_extended_image(img, x, y, w, h, k=0.1):  # img, x, y, w, h: from video
    if x - k * w > 0:
        start_x = int(x - k * w)
    else:
        start_x = x
    if y - k * h > 0:
        start_y = int(y - k * h)
    else:
        start_y = y

    end_x = int(x + (1 + k) * w)
    end_y = int(y + (1 + k) * h)

    face_image = img[start_y:end_y, start_x:end_x]
    face_image = tf.image.resize(face_image, [250, 250])  # Check the Shape
    # shape from (250, 250, 3) to (1, 250, 250, 3)
    face_image = np.expand_dims(face_image, axis=0)
    return face_image


# Streaming or Video File
video_capture = cv2.VideoCapture()

if not video_capture.isOpened():
    print("Unable to access the camera, Replaces with saved video files")
    video_capture = cv2.VideoCapture(VIDEO_FILE)  # Instead of Streaming
else:
    print("Access to the camera was successfully obtained")
    video_capture = cv2.VideoCapture(0)  # Check the Activated Cam Number

print("Streaming started - to quit press ESC")


# Loop for Real-Time
while True:

    ret, frame = video_capture.read()
    if not ret:
        print("Can't Receive Frame, Exit Streaming")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Loop
    for (x, y, w, h) in faces:
        face_image = get_extended_image(frame, x, y, w, h, 0.5)



        # <de-identification>
        # 1. gaussian_filter() -> face_blurred
        face_blurred = frame  # add
        face_blurred[y:y+h, x:x+w] = gaussian_filter(face_blurred[y:y+h, x:x+w], 10)  # add

        # 2. cv2.blur() -> cv2_blur (expected)
        cv2_blur = frame  # add
        cv2_blur[y:y + h, x:x + w] = cv2.blur(cv2_blur[y:y+h, x:x+w])  # add

        # 3. cv2.GaussianBlur(src, (0, 0), sigma) -> cv2_Gau_blur (expected)





        # Predict Result
        result = face_classifier.predict(face_image)
        prediction = class_names[np.array(
            result[0]).argmax(axis=0)]  # predicted class
        confidence = np.array(result[0]).max(axis=0)  # degree of confidence

        if prediction == 'me':
            color = GREEN
        else:
            color = RED

        # Draw a rectangle around the face
        cv2.rectangle(face_blurred,  # frame: 원본 / face_blurred: 변경
                      (x, y),  # start_point
                      (x+w, y+h),  # end_point
                      color,
                      2)  # thickness in px
        cv2.putText(frame,
                    # text to put
                    "{:6} - {:.2f}%".format(prediction, confidence*100),
                    (x, y),
                    cv2.FONT_HERSHEY_PLAIN,  # font
                    2,  # fontScale
                    color,
                    2)  # thickness in px

    # Display
    cv2.imshow("Real-time Face detector", frame)

    # Exit with ESC
    key = cv2.waitKey(1)
    if key % 256 == 27:  # ESC code
        break

video_capture.release()
cv2.destroyAllWindows()
