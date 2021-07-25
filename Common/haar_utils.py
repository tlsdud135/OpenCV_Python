import cv2

def preprocessing(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.equalizeHist(gray)
    return gray