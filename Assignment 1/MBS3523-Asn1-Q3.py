import cv2
import time
import numpy as np
import random

cap = cv2.VideoCapture('MBS3523 Resources/Video.mp4')

car_cascade = cv2.CascadeClassifier('MBS3523 Resources/cars.xml')
face=cv2.CascadeClassifier('MBS3523 Resources/fullbody.xml')

while True:
    ret, frames = cap.read()
    gray=cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    a = random.randint(0, 256)
    b = random.randint(0, 256)
    c = random.randint(0, 256)
    d = random.randint(0, 256)
    e = random.randint(0, 256)
    f = random.randint(0, 256)
    cars = car_cascade.detectMultiScale(gray, 1.99, 2)
    hum = face.detectMultiScale(gray, 1.07, 4)
    for (x, y, w, h) in hum:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (a, b, c), 1)
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (d, e, f), 1)
        cv2.imshow('Detection', frames)
    if cv2.waitKey(10) == 27:
        break
cap.release()
cv2.destroyAllWindows()
