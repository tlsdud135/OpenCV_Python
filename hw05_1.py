import numpy as np, cv2

cap = cv2.VideoCapture('images/a1.mp4')

images = []
t = 0
STEP = 20
while True:
    retval, frame = cap.read()
    if not retval:
        break

    t += 1
    if t % STEP == 0:
        img = cv2.resize(frame, dsize=(640, 480))
        images.append(img)

    cv2.imshow('img', frame)
    key = cv2.waitKey(25)

stitcher = cv2.Stitcher.create()
status, dst = stitcher.stitch(images)

if status == cv2.STITCHER_OK:
    cv2.imshow('dst', dst)
    cv2.waitKey()

cap.release()
