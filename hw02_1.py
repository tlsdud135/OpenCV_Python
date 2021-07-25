import cv2

def onThreshhold(value):
    th[0]=cv2.getTrackbarPos("V_min",title)
    th[1] = cv2.getTrackbarPos("V_max", title)
    h, s, v = cv2.split(hsv_img)
    r, v = cv2.threshold(v, th[1], 255, cv2.THRESH_TOZERO_INV)
    r, v=cv2.threshold(v, th[0], 255, cv2.THRESH_TOZERO)
    img = cv2.merge([h, s, v])
    image = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow(title, image)

def V_min(value):  # 트랙바 콜백 함수
    h,s,v=cv2.split(hsv_img)
    r, v=cv2.threshold(v,value,255,cv2.THRESH_TOZERO)
    img=cv2.merge([h,s,v])
    image = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow(title, image)    # 변환

def V_max(value):  # 트랙바 콜백 함수
    h, s, v = cv2.split(hsv_img)
    r, v=cv2.threshold(v,value,255,cv2.THRESH_TOZERO_INV)
    img = cv2.merge([h, s, v])
    image = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow(title, image)

def V_min2(value):  # 트랙바 콜백 함수
    h,s,v=cv2.split(hsv_img2)
    r, v=cv2.threshold(v,value,255,cv2.THRESH_TOZERO)
    img=cv2.merge([h,s,v])
    image = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow(title2, image)    # 변환

def V_max2(value):  # 트랙바 콜백 함수
    h, s, v = cv2.split(hsv_img2)
    r, v=cv2.threshold(v,value,255,cv2.THRESH_TOZERO_INV)
    img = cv2.merge([h, s, v])
    image = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow(title2, image)

def V_min3(value):  # 트랙바 콜백 함수
    h,s,v=cv2.split(hsv_img3)
    r, v=cv2.threshold(v,value,255,cv2.THRESH_TOZERO)
    img=cv2.merge([h,s,v])
    image = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow(title3, image)    # 변환

def V_max3(value):  # 트랙바 콜백 함수
    h, s, v = cv2.split(hsv_img3)
    r, v=cv2.threshold(v,value,255,cv2.THRESH_TOZERO_INV)
    img = cv2.merge([h, s, v])
    image = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow(title3, image)


image=cv2.imread("images/flower.png",cv2.IMREAD_COLOR)
image2=cv2.imread("images/blue.jpg",cv2.IMREAD_COLOR)
image3=cv2.imread("images/Illust.jpg",cv2.IMREAD_COLOR)
if image is None:raise Exception("영상파일 읽기 오류")
if image2 is None:raise Exception("영상파일 읽기 오류2")
if image3 is None:raise Exception("영상파일 읽기 오류3")

title = "img"
title2 = "img2"
title3 = "img3"
hsv_img=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
hsv_img2=cv2.cvtColor(image2,cv2.COLOR_BGR2HSV)
hsv_img3=cv2.cvtColor(image3,cv2.COLOR_BGR2HSV)

cv2.imshow(title, image)
#cv2.imshow(title2, image2)
#cv2.imshow(title3, image3)

th=[0,255]
cv2.createTrackbar('V_min', title, th[0], 255, onThreshhold)  # 트랙바 콜백 함수 등록
cv2.createTrackbar('V_max', title, th[1], 255, onThreshhold)

cv2.createTrackbar('V_min', title2, 0, 255, V_min2)  # 트랙바 콜백 함수 등록
cv2.createTrackbar('V_max', title2, 255, 255, V_max2)

cv2.createTrackbar('V_min', title3, 0, 255, V_min3)  # 트랙바 콜백 함수 등록
cv2.createTrackbar('V_max', title3, 255, 255, V_max3)

cv2.waitKey()
cv2.destroyAllWindows()