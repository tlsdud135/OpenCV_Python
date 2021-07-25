import numpy as np, cv2


def cir(event, x, y, flags, param):
    global pts1, pts2, a, i
    if event == cv2.EVENT_LBUTTONUP:

        image2 = np.zeros_like(image, np.uint8)
        cv2.circle(image1, (x, y), 10, (0, 255, 0), 1)
        cv2.circle(image2, (x, y), 10, 80, -1)
        a.append((x, y))
        cv2.add(image1, image2, image1)
        i=i + 1
        cv2.imshow("select rect", image1)
    if i == 4:
        pts1 = np.float32([a[0], a[3], a[2], a[1]])
        pts2 = np.float32([(0, 0), (400, 0), (400, 500), (0, 500)])
        cv2.polylines(image1, [pts1.astype(int)], True, (0, 255, 0), 1)
        cv2.imshow("select rect", image1)
        warp(np.copy(image))




def warp(img):

    perspect_mat = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, perspect_mat, (400, 500), cv2.INTER_CUBIC)
    img2 = np.zeros_like(dst, np.uint8)
    logo = cv2.imread('images/logo.png')
    if image is None: raise Exception("영상 파일을 읽기 에러")
    logo=cv2.resize(logo,(100,100),interpolation=cv2.INTER_LINEAR)
    img2[0:100,0:100]=logo

    bw=cv2.threshold(img2,0,255,cv2.THRESH_BINARY)[1]
    bw=cv2.bitwise_not(bw)
    bw=cv2.bitwise_and(bw,dst)

    dst2 = cv2.add(bw, img2)
    cv2.imshow("perspective transform", dst2)


image = cv2.imread('images/book.png')

if image is None: raise Exception("영상 파일을 읽기 에러")

image1=image.copy()
cv2.imshow("select rect", image1)
i=0
a=[]
cv2.setMouseCallback("select rect", cir, 0)



cv2.waitKey(0)
