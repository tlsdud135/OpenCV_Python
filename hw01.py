import cv2
import numpy as np


def red_bar(value):  # 트랙바 콜백 함수-red
    global red  # 전역 변수 참조
    red = value
    image[:] = (blue, green, red)  # 색 설정
    cv2.imshow(title, image)    # 변환

def green_bar(value):  # 트랙바 콜백 함수-green
    global green
    green = value
    image[:] = (blue, green, red)
    cv2.imshow(title, image)

def blue_bar(value):  # 트랙바 콜백 함수-blue
    global blue
    blue = value
    image[:] = (blue, green, red)
    cv2.imshow(title, image)

def onMouse(event, x, y, flags, param):
    global pt
    if event == cv2.EVENT_LBUTTONDOWN : #왼쪽 마우스버튼 눌렀을경우
        if flags & cv2.EVENT_FLAG_SHIFTKEY: #shift 활용
            if pt[0] < 0:
                pt = (x, y)                         # 시작 좌표 지정
            else:
                cv2.line(image, pt, (x, y), (255,0,0), 1)   #파란색 직선
                cv2.imshow(title, image)
                pt = (-1, -1)
        elif flags & cv2.EVENT_FLAG_ALTKEY: #alt 활용
            if pt[0] < 0:
                pt = (x, y)                         # 시작 좌표 지정
            else:
                cv2.rectangle(image, pt, (x, y), (255,255,0), 3)    #하늘색 사각형
                cv2.imshow(title, image)
                pt = (-1, -1)
        elif flags & cv2.EVENT_FLAG_CTRLKEY:    #ctrl활용
            if pt[0] < 0:
                pt = (x, y)                         # 시작 좌표 지정
                cv2.circle(image, pt, 1, 0, 2)  # 타원의 중심점(2화소 원) 표시
                cv2.imshow(title, image)
            else:
                dx, dy = pt[0] - x, pt[1] - y  # 두 좌표 간의 거리
                radius = int(np.sqrt(dx * dx + dy * dy))
                cv2.circle(image, pt, radius, (0, 0, 255), 2)   #빨간색 원
                cv2.imshow(title, image)
                pt = (-1, -1)
    elif event == cv2.EVENT_RBUTTONDOWN:    #마우스 오른쪽 버튼 눌렀을 경우
        if flags & cv2.EVENT_FLAG_ALTKEY:   #alt활용
            if pt[0] < 0:
                pt = (x, y)                         # 시작 좌표 지정
            else:
                cv2.rectangle(image, pt, (x, y), (0,255,255), cv2.FILLED)   #내부 칠한 노란색 사각형
                cv2.imshow(title, image)
                pt = (-1, -1)
        elif flags & cv2.EVENT_FLAG_CTRLKEY:    #ctrl활용
            if pt[0] < 0:
                pt = (x, y)                         # 시작 좌표 지정
                cv2.circle(image, pt, 1, 0, 2)  # 타원의 중심점(2화소 원) 표시
                cv2.imshow(title, image)
            else:
                dx, dy = pt[0] - x, pt[1] - y  # 두 좌표 간의 거리
                radius = int(np.sqrt(dx * dx + dy * dy))
                cv2.circle(image, pt, radius, (0, 255, 0), cv2.FILLED)  # 내부 칠한 초록색 원
                cv2.imshow(title, image)
                pt = (-1, -1)


image = np.zeros((400, 600, 3), np.uint8)   # 배경
red,green,blue=255,255,255  #색 초기값
image[:] = (blue, green, red)

title = "DrawBoard"
cv2.imshow(title, image)

cv2.createTrackbar('red', title, red, 255, red_bar)  # 트랙바 콜백 함수 등록
cv2.createTrackbar('green', title, green, 255, green_bar)
cv2.createTrackbar('blue', title, blue, 255, blue_bar)

pt = (-1, -1)
cv2.setMouseCallback(title, onMouse)    #마우스 콜백 함수 등록

cv2.waitKey(0)