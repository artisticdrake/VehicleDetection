import cv2
import numpy as np
from time import sleep

# Constants
min_width = 80  # Minimum width of the rectangle
min_height = 80  # Minimum height of the rectangle
offset = 6  # Allowed error between pixels
line_position = 200  # Position of the counting line
skew_position = 550
fps = 60  # Video Frames per second
detected = []
cars = 0

# Function to get center of rectangle
def get_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Video capture
cap = cv2.VideoCapture('output2.mp4')
subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame1 = cap.read()
    time = float(1 / fps)
    sleep(time)
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = subtractor.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (10, line_position), (550, skew_position), (255, 127, 0), 3)  # Vertical line on the right side
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_contour = (w >= min_width) and (h >= min_height)
        if not validate_contour:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = get_center(x, y, w, h)
        detected.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        # Color detection
        roi = frame1[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([0, 0, 100])
        upper_yellow = np.array([50, 10, 100])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        if cv2.countNonZero(mask) > 0:
            cv2.putText(frame1, "Blinker Detected", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        for (x, y) in detected:
            #print("x,y: ",x,",",y)
            #print("\nline,skew: ", line_position, "," , skew_position)
            if x < (line_position + offset) and y > (line_position - offset):
                
                cars += 1
                cv2.line(frame1, (10, line_position), (550, skew_position), (0, 127, 255), 3)  # Change color of line when car detected
                detected.remove((x, y))
                print("Vehicle is detected: " + str(cars))

    cv2.putText(frame1, "VEHICLE COUNT: " + str(cars), (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    cv2.imshow("Original Video", frame1)
    cv2.imshow("Detection", dilated)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
