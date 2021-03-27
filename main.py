import cv2
import sys
import numpy as np

cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        kernel_width = (w // 7) | 1
        kernel_height = (h // 7) | 1
        start_x, start_y = (x, y)
        end_x, end_y = (x+w, y+h)
        
        face = frame[start_y: end_y, start_x: end_x]
        for i in range(1, 20):
            face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 2)
        
        frame[start_y: end_y, start_x: end_x] = face

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()