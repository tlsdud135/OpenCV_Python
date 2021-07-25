import numpy as np, cv2

def color_candidate_img(image, center):
    h, w = image.shape[:2]
    fill = np.zeros((h + 2, w + 2), np.uint8)           # 채움 영역
    dif1, dif2 = (25, 25, 25), (25, 25, 25)             # 채움 색상 범위
    flags = 4 + 0xff00 + cv2.FLOODFILL_FIXED_RANGE                                # 채움 방향
    flags += cv2.FLOODFILL_MASK_ONLY

    # 후보 영역을 유사 컬러로 채우기
    pts = np.random.randint( -15, 15, (20,2) )          # 20개 좌표 생성
    pts = pts + center
    for x, y in pts:                                 # 랜덤 좌표 평행 이동
        if 0 <= x < w and 0 <= y < h:
            _, _, fill, _ = cv2.floodFill(image, fill, (x,y), 255, dif1, dif2, flags)

    # 이진화 및 외곽영역 추출후 회전 사각형 검출
    return cv2.threshold(fill, 120, 255, cv2.THRESH_BINARY)[1]

def rotate_plate(image, rect):
    center, (w, h), angle = rect       # 중심점, 크기, 회전 각도
    if w < h :                         # 세로가 긴 영역이면
        w, h = h, w                    # 가로와 세로 맞바꿈
        angle += 90                    # 회전 각도 조정

    size = image.shape[1::-1]            # 행태와 크기는 역순
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)  # 회전 행렬 계산
    rot_img= cv2.warpAffine(image, rot_mat, size, cv2.INTER_CUBIC)  # 회전 변환

    crop_img = cv2.getRectSubPix(rot_img, (w, h), center)  # 후보영역 가져오기
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(crop_img, (144, 28))


def preprocessing(car_no):
    image = cv2.imread("images/car2/%s.jpg" % car_no, cv2.IMREAD_COLOR)
    if image is None: return None, None

    kernel = np.ones((5, 13), np.uint8)  # 닫힘 연산 마스크
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 명암도 영상 변환
    gray = cv2.blur(gray, (5, 5))  # 블러링
    gray = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)  # 소벨 에지 검출

    th_img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]  # 이진화 수행
    morph = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, kernel, iterations=3)

    # cv2.imshow("th_img", th_img); cv2.imshow("morph", morph)
    return image, morph


def verify_aspect_size(size):
    w, h = size
    if h == 0 or w == 0: return False

    aspect = h / w if h > w else w / h  # 종횡비 계산

    chk1 = 3000 < (h * w) < 12000  # 번호판 넓이 조건
    chk2 = 2.0 < aspect < 6.5  # 번호판 종횡비 조건
    return (chk1 and chk2)


def find_candidates(image):
    results = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = results[0] if int(cv2.__version__[0]) >= 4 else results[1]

    rects = [cv2.minAreaRect(c) for c in contours]  # 외곽 최소 영역
    candidates = [(tuple(map(int, center)), tuple(map(int, size)), angle)
                  for center, size, angle in rects if verify_aspect_size(size)]

    return candidates



# 숫자 및 문자 영상 학습
def kNN_train(train_fname, K, nclass, nsample):
    size = (40, 40)  # 숫자 영상 크기
    train_img = cv2.imread(train_fname, cv2.IMREAD_GRAYSCALE)  # 학습 영상 적재
    h, w = train_img.shape[:2]
    dy = h % size[1]// 2
    dx = w % size[0]// 2
    train_img = train_img[dy:h-dy-1, dx:w-dx-1]             # 학습 영상 여백 제거
    cv2.threshold(train_img, 32, 255, cv2.THRESH_BINARY, train_img)

    cells = [np.hsplit(row, nsample) for row in np.vsplit(train_img, nclass)]
    nums = [find_number(c) for c in np.reshape(cells, (-1, 40,40))]
    trainData = np.array([place_middle(n, size) for n in nums])
    labels = np.array([i for i in range(nclass) for j in range(nsample)], np.float32)

    knn = cv2.ml.KNearest_create()
    knn.train(trainData, cv2.ml.ROW_SAMPLE, labels)  # k-NN 학습 수행
    return knn

# 번호판 영상 전처리
def preprocessing_plate(plate_img):
    plate_img = cv2.resize(plate_img, (180, 35))            # 번호판 영상 크기 정규화
    flag = cv2.THRESH_BINARY | cv2.THRESH_OTSU              # 이진화 방법
    cv2.threshold(plate_img, 32, 255, flag, plate_img)      # 이진화

    h, w = plate_img.shape[:2]
    dx, dy = (6, 3)
    ret_img= plate_img[dy:h-dy, dx:w-dx]                    # 여백 제거
    return ret_img

# 숫자 및 문자 객체 검색
def find_objects(sub_mat):
    results = cv2.findContours(sub_mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = results[0] if int(cv2.__version__[0]) >= 4 else results[1]

    rois = [cv2.boundingRect(contour) for contour in contours]
    rois = [(x, y, w, h, w*h) for x,y,w,h in rois if w / h < 2.5]

    text_rois = [(x, y, x+w, y+h) for x, y, w, h, a in rois if 45 < x < 80 and a > 60]
    num_rois  = [(x, y, w, h) for x, y, w, h, a in rois  if not(45 < x < 80) and a > 150]

    if text_rois:                         # 분리된 문자 영역 누적
        # pts= cv2.sort(np.array(text_rois), cv2.SORT_EVERY_COLUMN)  # 열단위 오름차순
        pts= np.sort(text_rois, axis=0)             # y 방향 정렬
        x0, y0 = pts[ 0, 0:2]                  # 시작좌표 중 최소인 x, y 좌표
        x1, y1 = pts[-1, 2:]                         # 종료좌표 중 최대인 x, y 좌표
        w, h = x1-x0, y1-y0                             # 너비, 높이 계산
        num_rois.append((x0, y0, w, h))             # 문자 영역 구성 및 저장

    return num_rois

# 검출 객체 영상의 숫자 및 문자 인식
def classify_numbers(cells, nknn, tknn, K1, K2, object_rois):
    if len(cells) != 7:
        print("검출된 숫자(문자)가 7개가 아닙니다.")
        return

    texts  = "가나다라마거너더러머버서어저고노도로모보" \
             "소오조구누두루무부수우주아바사자바하허호"

    numbers = [find_number(cell) for cell in cells]
    datas = [place_middle(num, (40,40)) for num in numbers]
    datas = np.reshape(datas, (len(datas), -1))

    idx = np.argsort(object_rois, axis=0).T[0]
    text = datas[idx[2]].reshape(1,-1)

    _, resp1, _, _ = nknn.findNearest(datas, K1)  # 숫자 k-NN 분류 수행
    _, [[resp2]], _, _ = tknn.findNearest(text, K2)  # 문자 k-NN 분류 수행

    resp1 = resp1.flatten().astype('int')
    results = resp1[idx].astype(str)
    results[2] = texts[int(resp2)]

    print("정렬 인덱스:", idx)
    print("숫자 분류 결과:", resp1)
    print("문자 분류 결과:", int(resp2))
    print("분류 결과: ", ' '.join(results))


def find_value_position(img, direct):
    project = cv2.reduce(img, direct, cv2.REDUCE_AVG).ravel()
    p0, p1 = -1, -1                                                 # 초기값
    len = project.shape[0]                                   # 전체 길이
    for i in range(len):
        if p0 < 0 and project[i] < 250: p0 = i
        if p1 < 0 and project[len-i-1] < 250 : p1 = len-i-1
    return p0, p1

def place_middle(number, new_size):
    h, w = number.shape[:2]
    big = max(h, w)
    square = np.full((big, big), 255, np.float32)  # 실수 자료형

    dx, dy = np.subtract(big, (w,h))//2
    square[dy:dy + h, dx:dx + w] = number
    return cv2.resize(square, new_size).flatten()  # 크기변경 및 벡터변환 후 반환

def find_number(part):
    x0, x1 = find_value_position(part, 0)  # 수직 투영
    y0, y1 = find_value_position(part, 1)  # 수평 투영
    return part[y0:y1, x0:x1]



car_no = input("자동차 영상 이름: ")
image, morph = preprocessing(car_no)                               # 전처리
candidates = find_candidates(morph)                        # 번호판 후보 영역 검색

fills = [color_candidate_img(image, size) for size, _, _ in candidates]
new_candis = [find_candidates(fill) for fill in fills]
new_candis = [cand[0] for cand in new_candis if cand]
candidate_imgs = [rotate_plate(image, cand) for cand in new_candis]

svm = cv2.ml.SVM_load("SVMTrain.xml")                  # 학습된 데이터 적재
rows = np.reshape(candidate_imgs, (len(candidate_imgs), -1))    # 1행 데이터들로 변환
_, results = svm.predict(rows.astype("float32"))                # 분류 수행
correct = np.where(results == 1)[0]        # 1인 값의 위치 찾기

print('분류 결과:\n', results)
print('번호판 영상 인덱스:', correct )

for i, idx in enumerate(correct):
    #cv2.imshow("plate_" +str(i), candidate_imgs[idx])
    cv2.resizeWindow("plate image_" + str(i), (250,28))

for i, candi in enumerate(new_candis):
    color = (0, 255, 0) if i in correct else (0, 0, 255)
    cv2.polylines(image, [np.int32(cv2.boxPoints(candi))], True, color, 2)

print("번호판 검출완료") if len(correct)>0 else print("번호판 미검출")



plate_no = correct[0] if len(correct)>0 else -1

K1, K2 = 10, 10
nknn = kNN_train("images/train_numbers.png", K1, 10, 20) # 숫자 학습
tknn = kNN_train("images/train_texts.png", K2, 40, 20)   # 문자 학습

if plate_no >= 0:
    plate_img = preprocessing_plate(candidate_imgs[plate_no])   # 번호판 영상 전처리
    cells_roi = find_objects(cv2.bitwise_not(plate_img))
    cells = [plate_img[y:y+h, x:x+w] for x,y,w,h in cells_roi]

    classify_numbers(cells, nknn, tknn, K1, K2, cells_roi)      # 숫자 객체 분류

    pts = np.int32(cv2.boxPoints(new_candis[plate_no]))
    cv2.polylines(image, [pts], True,  (0, 255, 0), 2)

    color_plate = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)  # 컬러 번호판 영상
    for x,y, w, h in cells_roi:
        cv2.rectangle(color_plate, (x,y), (x+w,y+h), (0, 0, 255), 1)        # 번호판에 사각형 그리기

    h,w  = color_plate.shape[:2]
    image[0:h, 0:w] = color_plate
else:
    print("번호판 미검출")

cv2.imshow("image", image)
cv2.waitKey(0)