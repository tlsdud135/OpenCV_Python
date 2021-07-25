import cv2
import numpy as np
import pafy
from PIL import ImageFont, ImageDraw, Image

classes = []
with open("./dnn/coco_k.names", "rt", encoding="UTF8") as f:  # yolo v2
    classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

font = ImageFont.truetype("fonts/gulim.ttc", 20)

# 실습
model = cv2.dnn.readNet("./dnn/yolov3.weights", "./dnn/yolov3.cfg")
layer_names = model.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]

CONF_THR = 0.5

video = cv2.VideoCapture('images/yolomovie.mp4')

while True:
    ret, frame = video.read()
    if not ret: break

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # 처리 가능 형태로 변경
    model.setInput(blob)
    output = model.forward(output_layers)

    h, w = frame.shape[0:2]
    img = cv2.resize(frame, dsize=(int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
    ih = int(h / 2)
    iw = int(w / 2)

    class_ids = []
    confidences = []
    boxes = []
    for out in output:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]

            if conf > CONF_THR:
                center_x = int(detection[0] * iw)
                center_y = int(detection[1] * ih)
                w = int(detection[2] * iw)
                h = int(detection[3] * ih)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(conf))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            conf = confidences[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            #cv2.putText(img, '{}: {:.2f}'.format(label, conf), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x, y + 50), '{}: {:.2f}'.format(label, conf), font=font, fill=(55, 55, 0))  # fill=tuple(color))
            img = np.array(img_pil)

    cv2.imshow('frame', img)

    #cv2.imshow('frame', frame)
    key = cv2.waitKey(3)
    if key == 27: break

cv2.destroyAllWindows()
