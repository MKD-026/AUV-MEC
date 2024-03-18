import cv2
import numpy as np
from ultralytics import YOLO
import time

def delta_xy(imgcenter_x, imgcenter_y, center_x, center_y, classID):
    delta_x = imgcenter_x - center_x
    delta_y = imgcenter_y - center_y

    rounded_delta_x = round(delta_x, 3)
    rounded_delta_y = round(delta_y, 3)

    if(delta_x > 0.0):
        direction = "L"
        # print("Left", class_list[int(clsID)], rounded_delta_x, rounded_delta_y)
    else:
        direction = "R"
        # print("Right", class_list[int(clsID)], rounded_delta_x, rounded_delta_y)

    return direction, (rounded_delta_x, rounded_delta_y)

# load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

cap = cv2.VideoCapture(0)

#for video input
# cap = cv2.VideoCapture("videos/7.mp4")
if not cap.isOpened():
    print("Cannot open camera")
    exit()

prev_frame_time = 0
new_frame_time = 0
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    image_height, image_width = frame.shape[0], frame.shape[1]
    imgcenter_x = image_width / 2
    imgcenter_y = image_height / 2

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    new_frame_time = time.time()
    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.65, save=False)

    DP = detect_params[0].numpy()
    # print(DP)

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 3)
            center_x = (bb[0] + bb[2]) / 2
            center_y = (bb[1] + bb[3]) / 2

            cv2.line(frame, (int(center_x), int(center_y)), (int(imgcenter_x), int(imgcenter_y)), (255, 255, 255), 5)
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(imgcenter_x), int(imgcenter_y)), 5, (0, 0, 255), -1)

            direction, diff = delta_xy(imgcenter_x, imgcenter_y, center_x, center_y, clsID)

            font = cv2.FONT_HERSHEY_COMPLEX

            cv2.putText(
                frame, str(diff),
                (int(center_x + 25), int(center_y)), font, 1, (0, 255, 0), 2
            )

            cv2.putText(
                frame, direction,
                (int(center_x - 20), int(center_y)), font, 1, (0, 255, 0), 2
            )

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    fps = int(fps)
    fps = str(fps)
    cv2.putText(
        frame, fps,
        (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
    )
    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
