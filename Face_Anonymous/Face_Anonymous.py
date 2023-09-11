import cv2
import mediapipe as mp


from Help_Functions import resize_video

max_width = 800

def process_img(img, face_detection):
    (h, w, _) = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)

            img[y1:y1 + y2, x1:x1 + x2, :] = cv2.blur(img[y1:y1 + y2, x1:x1 + x2, :], (50, 50))

    return img


mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()

        if not ret:
            break

        if max_width is not None:
            video_width, video_height = resize_video(frame.shape[1], frame.shape[0], max_width)

        processed_frame = process_img(frame, face_detection)

        cv2.imshow('Detecting faces', processed_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print('Finished')
