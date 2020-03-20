import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import transforms as transforms
from skimage import io
from skimage.transform import resize
from EdgeCNN import *
import cv2 as cv
import dlib
import time
from EdgeCNN import *


transform_test = transforms.Compose([ 
    transforms.ToTensor(),
])

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return x, y, w, h

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return x - x_off, x + width + x_off, y - y_off, y + height + y_off

def get_color(emotion, prob):
    if emotion.lower() == 'angry':
        color = (0, 0, 255)
    elif emotion.lower() == 'disgust':
        color = (255, 0, 0)
    elif emotion.lower() == 'fear':
        color = (0, 255, 255)
    elif emotion.lower() == 'happy':
        color = (255, 255, 0)
    elif emotion.lower() == 'sad':
        color = (255, 255, 255)
    elif emotion.lower() == 'surprise':
        color = (255, 0, 255)
    else:
        color = (0, 255, 0)
    return color

def draw_bounding_box(image, coordinates, color):
    x, y, w, h = coordinates
    cv.rectangle(image, (x, y), (x + w, y + h), color, 3)
    
def draw_text(image, coordinates, text, color, x_offset=0, y_offset=0,
              font_scale=1, thickness=2):
    x, y = coordinates[:2]
    cv.putText(image, text, (x + x_offset, y + y_offset),
                cv.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv.LINE_AA)

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


cap = cv.VideoCapture(0) # get frames from webcam
detector = dlib.get_frontal_face_detector() # get frontal face detector from dlib
fps_vector = []
face_vetor = []

net = EdgeCNN() # get network architecture
checkpoint = torch.load(os.path.join('/Users/pauldamsa/Desktop/licenta/PyImageSearch/face-detection/raspberrypi/RAF_EdgeCNN/PrivateTest_model.t7'),
                        map_location=lambda storage, loc: storage)
net.load_state_dict(checkpoint['net'])
net.cpu()
net.eval()
print('MODEL ACCURACY: %.2f'% checkpoint['best_PrivateTest_acc'])
cap = cv.VideoCapture(0)

while(1):
    start = time.time()
    # get a frame
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    # show a frame
    frame = frame[100:100 + width, :]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
    faces = detector(gray, 1)
    
    for rect in faces:
        (x, y, w, h) = rect_to_bb(rect)
        x1, x2, y1, y2 = apply_offsets((x, y, w, h), (10, 10))
        gray_face = gray[y1:y2, x1:x2]
       
        img = cv.resize(gray_face, (44, 44))
        
        img = Image.fromarray(img)
        inputs = transform_test(img)

        c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)
        
        outputs = net.forward(inputs)
        
        score = F.softmax(outputs)
        
        _, predicted = torch.max(outputs.data,1)

        emotion = class_names[int(predicted.cpu().numpy())]
        prob = max(score.data.cpu().numpy())
    
        color = get_color(emotion, prob)
        
        text = emotion + ' ' + str(round(max(prob), 5) * 100)
        print(text)
        draw_bounding_box(image=frame, coordinates=(x1, y1, x2 - x1, y2 - y1), color=color)
        draw_text(image=frame, coordinates=(x1, y1, x2 - x1, y2 - y1), color=color, text=emotion)
    cv.imshow("FACIAL EXPRESSION RECOGNITION VIDEO STREAM", frame) 

    if cv.waitKey(1) == ord('q'):
        break

    end = time.time()
    seconds = end - start
    fps = 1.0 / seconds
    fps_vector.append(fps)
    face_vetor.append(len(faces))
    print('faces: %.2f' % len(faces))
    print("fps: %.2f" % fps)

average_fps = sum(fps_vector) / len(fps_vector)
averate_faces = sum(face_vetor) / len(face_vetor)

print("MEAN FPS DLIB: %.2f" % average_fps)
print("MEAN DETECTED FACES DLIB: %.2f" % averate_faces)

cap.release()
cv.destroyAllWindows()














