import cv2
import numpy as np
from time import sleep

# Constants for the first ROI
min_width_1 = 80  # Minimum width of the rectangle
min_height_1 = 80  # Minimum height of the rectangle
offset_1 = 6  # Allowed error between pixels
line_position_1 = 850  # Position of the counting line
detected_1 = []
cars_1 = 0

# Constants for the second ROI
min_width_2 = 80  # Minimum width of the rectangle
min_height_2 = 80  # Minimum height of the rectangle
offset_2 = 6  # Allowed error between pixels
line_position_2 = 300  # Position of the counting line
detected_2 = []
cars_2 = 0

# Function to get center of rectangle
def get_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Video capture
cap = cv2.VideoCapture('output.mp4')
subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame1 = cap.read()
    if not ret:
        break
    time = float(1 / 60)
    sleep(time)
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = subtractor.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # First ROI
    cv2.line(frame1, (line_position_1, 25), (line_position_1, 650), (255, 127, 0), 3)  # Vertical line on the right side
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_contour = (w >= min_width_1) and (h >= min_height_1)
        if not validate_contour:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = get_center(x, y, w, h)
        detected_1.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in detected_1:
            if x < (line_position_1 + offset_1) and x > (line_position_1 - offset_1):
                cars_1 += 1
                cv2.line(frame1, (line_position_1, 25), (line_position_1, 650), (0, 127, 255), 3)  # Change color of line when car detected
                detected_1.remove((x, y))
                print("vehicle is detected in ROI 1: " + str(cars_1))

    # Second ROI
    cv2.line(frame1, (0,line_position_2), ( 300,line_position_2), (255, 127, 0), 3)  # Vertical line on the right side
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_contour = (w >= min_width_2) and (h >= min_height_2)
        if not validate_contour:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = get_center(x, y, w, h)
        detected_2.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in detected_2:
            if y < (line_position_2 + offset_2) and y > (line_position_2 - offset_2):
                cars_2 += 1
                cv2.line(frame1, (0,line_position_2), ( 300,line_position_2), (0, 127, 255), 3)  # Change color of line when car detected
                detected_2.remove((x, y))
                print("vehicle is detected in ROI 2: " + str(cars_2))

    cv2.putText(frame1, "VEHICLE COUNT (Incoming): " + str(cars_1), (725, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    cv2.putText(frame1, "VEHICLE COUNT (Outgoing): " + str(cars_2), (725, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow("Original Video", frame1)
    #cv2.imshow("Detection", dilated)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
