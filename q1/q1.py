import cv2
import numpy as np

cap = cv2.VideoCapture("q1A.mp4")
colisao_ocorrida = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    conts_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    shape_bbox = None
    for cnt in conts_red:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            shape_bbox = cv2.boundingRect(cnt)

    conts_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area_barrier = 0
    barrier_bbox = None
    for cnt in conts_blue:
        area = cv2.contourArea(cnt)
        if area > max_area_barrier:
            max_area_barrier = area
            barrier_bbox = cv2.boundingRect(cnt)

    if shape_bbox is not None:
        sx, sy, sw, sh = shape_bbox
        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

    if shape_bbox is not None and barrier_bbox is not None:
        bx, by, bw, bh = barrier_bbox
        collision = (sx < bx + bw and sx + sw > bx and sy < by + bh and sy + sh > by)
        if collision:
            colisao_ocorrida = True
            cv2.putText(frame, "COLISAO DETECTADA", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else :
            if sx + sw < bx :
                cv2.putText(frame, "PASSOU BARREIRA", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            colisao_ocorrida = False
        
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
