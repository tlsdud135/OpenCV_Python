import cv2
import numpy as np
import pafy

url = "https://www.youtube.com/watch?v=CrxAJ2_i1wQ"

video = pafy.new(url)
best = video.getbest(preftype="mp4")
capture = cv2.VideoCapture(best.url)

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
frameWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frameWidth, frameHeight)
fps = int(capture.get(cv2.CAP_PROP_FPS))

f=0
while True:  # 무한 반복
    ret, frame = capture.read()  # 카메라 영상 받기
    if not ret: break
    if cv2.waitKey(1) >= 0: break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)


    if f==0:
        roi = cv2.selectROI(frame)
        f+=1
        roi_h = h[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    hist = cv2.calcHist([roi_h], [0], None, [60], [0, 180])
    backP = cv2.calcBackProject([hsv[:, :, 0].astype(np.float32)], [0], hist, [0, 180], scale=1.0)

    hist = cv2.sort(hist, cv2.SORT_EVERY_COLUMN + cv2.SORT_DESCENDING)
    k = 1
    T = hist[k][0] - 1  # threshold

    ret, dst = cv2.threshold(backP, T, 255, cv2.THRESH_BINARY)


    img3 = cv2.bitwise_and(frame, frame, mask=dst.astype(np.uint8))

    title = "View Frame from Youtube"
    cv2.imshow(title, img3)  # 윈도우에 영상 띄우기

capture.release()
