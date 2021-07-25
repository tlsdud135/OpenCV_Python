import cv2
import numpy as np

# Python으로 배우는 OpenCV 프로그래밍

vidio = cv2.VideoCapture("images/RSP.mp4")
if vidio.isOpened() == False:
    raise Exception("카메라 연결 안됨")

while True:
    ret, src = vidio.read()
    if not ret: break
    if cv2.waitKey(60) >= 0: break

    # 영역 추출
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lowerb = (0, 70, 0)
    upperb = (20, 180, 255)
    bImage = cv2.inRange(hsv, lowerb, upperb) # 특정 영역을 추출, 범위 내는 255, 그 외는 0으로

    #윤곽선
    mode   = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(bImage, mode, method)

    dst = src.copy()
    cnt = contours[0]
    cv2.drawContours(dst, [cnt], 0, (255,0,0), 2)

    #볼록다각형
    hull = cv2.convexHull(cnt, returnPoints=False)
    hull_points = cnt[hull[:,0]]
    dst2 = dst.copy()
    #cv2.drawContours(dst2, [hull_points], 0, (0,255,0), 2)

    #오목 부분 찾기
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    T = 50  # 임계값
    c=0
    if type(defects)==type(None):
        continue
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        dist = d / 256
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        if dist > T :
            c+=1

    if c<2:
        cv2.putText(dst2,'Rock',(dst2.shape[1]-150,50),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(255,0,255),4)
    elif c<5:
        cv2.putText(dst2, 'Scissors', (dst2.shape[1] - 220, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 0, 255), 4)
    else:
        cv2.putText(dst2, 'Paper', (dst2.shape[1] - 180, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 0, 255), 4)
    print(c)
    cv2.imshow('dst2',  dst2)

vidio.release()
