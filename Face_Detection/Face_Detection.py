import cv2
import numpy as np
import os

from Help_Functions import resize_video

detector = 'ssd'
max_width = 800


def detect_face_ssd(network, orig_frame, show_conf=True, conf_min=0.7):
    frame = orig_frame.copy()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 117.0, 123,0))
    network.setInput(blob)
    detections = network.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_min:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype('int')

            if (start_x < 0 or start_y < 0 or end_x > w or end_y > h):
                continue

            color = (0, 255, 0)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

            if show_conf:
                text_conf = '{:.2f}%'.format(confidence * 100)
                text_size = cv2.getTextSize(text_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (start_x, start_y - text_size[1] - 5), (start_x + text_size[0], start_y), color,
                              -1)
                cv2.putText(frame, text_conf, (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame


network = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

cam = cv2.VideoCapture(0)

while (True):
    ret, frame = cam.read()

    if max_width is not None:
        video_width, video_height = resize_video(frame.shape[1], frame.shape[0], max_width)

    processed_frame = detect_face_ssd(network, frame)

    cv2.imshow('Detecting faces', processed_frame)
    cv2.waitKey(1)


print('Finished')
cam.release()
cv2.destroyAllWindows()