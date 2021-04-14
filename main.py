import cv2
import sys
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat


cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# For the webcam output

width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)


with pyvirtualcam.Camera(width, height, fps, fmt=PixelFormat.BGR) as cam:
    print('Virtual camera device: ' + cam.device)
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
            start_x, start_y = (x-20, y-20)
            end_x, end_y = (x+w+20, y+h+20)
            
            face = frame[start_y: end_y, start_x: end_x]
            for i in range(1, 20):
                face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 2)
            
            frame[start_y: end_y, start_x: end_x] = face

        # Display the resulting frame
        cv2.imshow('Video', frame)
        cam.send(frame)
        cam.sleep_until_next_frame()
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# When everything is done, release the capture
video_capture.release()
out.release()
cv2.destroyAllWindows()