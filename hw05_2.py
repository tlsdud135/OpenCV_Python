import numpy as np, cv2
import matplotlib.pyplot as plt
import pafy

url = "https://youtu.be/IGJCJNeBit8"

video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv2.VideoCapture(best.url)

def frame_diff(prev_frame, cur_frame):
    return cv2.absdiff(prev_frame, cur_frame)

def get_frame(cap):
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        return []


fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
delay = int(1000/fps)

prev_frame = get_frame(cap)
cur_frame = get_frame(cap)

i = 0
nFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sum_diff = np.zeros(shape=(nFrame), dtype=np.int32)
cv2.imshow("Keyframe" + str(i), cur_frame)
while True:
    diff = frame_diff(prev_frame, cur_frame)
    sum_diff[i] = np.sum(diff)
    #print(sum_diff)

    if sum_diff[i] > 10000000 :
        cv2.imshow("Keyframe" + str(i), cur_frame)
        cv2.waitKey()

    cv2.imshow("Motion", diff)
    i = i + 1
    prev_frame = cur_frame
    cur_frame = get_frame(cap)

    if cur_frame == [] : break
    if cv2.waitKey(delay) & 0xff == 27 : break

x = np.arange(nFrame)
plt.figure(figsize=(5,3))
plt.plot(x, sum_diff, 'b-', lineWidth=2)
plt.title('Keyframe Selection')
plt.show()

cap.release()
