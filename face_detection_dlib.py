import cv2
import numpy as np
import dlib
import time

cap = cv2.VideoCapture(0) # get frames from webcam
detector = dlib.get_frontal_face_detector() # get frontal face detector from dlib
fps_vector = []
while True:
    start = time.time()
    ret, frame = cap.read() # read from the video stream

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # transform every frame in gray scale image
    
    faces = detector(gray)

    for face in faces:
        x, y = face.left(), face.top()
        w, h = face.right(), face.bottom()

        cv2.rectangle(frame,(x, y), (w, h),(0,255,0),3)

    # cv2.imshow("cropped gray image", gray[y:h, x:w])
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord('s'):
        break

    end = time.time()
    seconds = end - start
    fps = 1.0 / seconds
    fps_vector.append(fps)
    print('fps: %.2f' % fps)

average_fps = sum(fps_vector) / len(fps_vector)
print("MEAN FPS DLIB: %.2f" % average_fps)

cap.release()
cv2.destroyAllWindows()














