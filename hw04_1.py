import numpy as np, cv2
from Common.utils import put_string

#coin_utils
def make_coin_img(src, circles):
    coins = []
    for center, radius  in circles:
        r = radius * 3                      # 검출 동전 반지름의 3 배
        cen = (r // 2, r // 2)                                  # 마스크 중심
        mask = np.zeros((r, r, 3), np.uint8)                    # 마스크 행렬
        cv2.circle(mask, cen, radius, (255, 255, 255), cv2.FILLED)

        # 동전 영상 가져오기
        coin = cv2.getRectSubPix(src, (r, r), center)
        coin = cv2.bitwise_and(coin, mask)                      # 마스킹 처리
        coins.append(coin)                                      # 동전 영상 저장
        # cv2.imshow("mask_" + str(center) , mask)                    # 마스크 영상 보기
    return coins

def calc_histo_hue(coin):
    hsv = cv2.cvtColor(coin, cv2.COLOR_BGR2HSV)  # 컬러 공간 변환
    hsize, ranges = [32], [0, 180]         # 32개 막대, 화소값 0~180 범위
    hist = cv2.calcHist([hsv], [0], None, hsize, ranges)
    return hist.flatten()

def grouping(hists):
    ws = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3,
          4, 5, 6, 8, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0]        # 가중치 지정

    sim = np.multiply(hists, ws)
    similaritys = np.sum(sim, axis=1) / np.sum(hists, axis=1)

    groups = [1 if s > 1.2 else 0 for s in similaritys]

    # x = np.arange(len(ws))
    # plt.plot(x, ws, 'r'), plt.show(), plt.tight_layout()	 # 가중치 그래프 보기
    # for i, s in enumerate(similaritys):											# 그룹핑 결과 출력	#
    #     print("%d %5f %d" % (i, s, groups[i]))
    return groups

# 동전 인식 : 동전 종류 결정
def classify_coins(circles, groups):
    ncoins = [0] * 4
    g = np.full((2,70), -1, np.int32)
    g[0, 26:47], g[0, 47:50], g[0, 50:] = 0, 2, 3
    g[1, 36:44], g[1, 44:50], g[1, 50:] = 1, 2, 3

    for group, (_, radius) in zip(groups, circles):
        coin = g[group, radius]
        ncoins[coin] += 1

    return np.array(ncoins)


#coin_preprocess
def preprocessing(coin_no):  # 전처리 함수
    fname = "images/coin/{0:02d}.png".format(coin_no)
    image = cv2.imread(fname, cv2.IMREAD_COLOR)  # 영상읽기
    if image is None: return None, None  # 예외처리는 메인에서

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 명암도 영상 변환
    gray = cv2.GaussianBlur(gray, (7, 7), 2, 2)  # 블러링
    flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU  # 이진화 방법
    _, th_img = cv2.threshold(gray, 130, 255, flag)  # 이진화

    mask = np.ones((3, 3), np.uint8)
    th_img = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, mask)  # 열림 연산

    return image, th_img


def find_coins(image):
    results = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = results[0] if int(cv2.__version__[0]) >= 4 else results[1]

    # 반복문 방식
    # circles = []
    # for contour in contours:
    #     center, radius = cv2.minEnclosingCircle(contour)        # 외각을 둘러싸는 원 검출
    #     circle = (tuple(map(int, center)), int(radius))
    #     if radius>25: circles.append(circle)

    # 리스트 생성 방식
    circles = [cv2.minEnclosingCircle(c) for c in contours]  # 외각을 둘러싸는 원 검출
    circles = [(tuple(map(int, center)), int(radius))
               for center, radius in circles if radius > 25]
    return circles




coin_no = int(input("동전 영상 번호: "))
image, th_img = preprocessing(coin_no)                              # 전처리 수행
circles = find_coins(th_img)                     # 객체(회전사각형) 검출
coin_imgs = make_coin_img(image, circles)                  # 동전 영상 생성
coin_hists= [calc_histo_hue(coin) for coin in coin_imgs]   # 동전 영상 히스토그램

groups = grouping(coin_hists)                              # 동전 영상 그룹 분리
ncoins = classify_coins(circles, groups)                   # 동전 인식

coin_value = np.array([10, 50, 100, 500])                             # 동전 금액
for i in range(4):
    print("%3d원: %3d개" % (coin_value[i], ncoins[i]))

total = sum(coin_value * ncoins )           # 동전금액* 동전별 개수
str = "Total coin: {:,} Won".format(total)            # 계산된 금액 문자열
print(str)                                                 # 콘솔창에 출력
put_string(image, str, (650, 50), '', (0,230,0))

## 동전 객체에 정보(반지름, 금액) 표시
color = [(0, 0, 250), (255, 255, 0), (0, 250, 0), (250, 0, 255)]  # 동전별 색상
for i, (c, r) in enumerate(circles):
    cv2.circle(image, c, r, color[groups[i]], 2)
    put_string(image, i, (c[0] - 15, c[1] - 10), '', color[2])  # 검출 순번과 동전 반지 표시
    put_string(image, r, (c[0], c[1] + 15), '', color[3])

cv2.imshow("result image", image)
key = cv2.waitKey(0)  # 키 이벤트 대기
