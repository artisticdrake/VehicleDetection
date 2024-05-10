import cv2
import numpy as np
from time import sleep

# Constants for the first ROI
min_width_1 = 80  # Minimum width of the rectangle
min_height_1 = 80  # Minimum height of the rectangle
offset_1 = 6  # Allowed error between pixels
line_position_1 = 850  # Position of the counting line
detected_1 = []
prev_centers_1 = {}  # Dictionary to store previous centers
cars_1 = 0

# Constants for the second ROI
min_width_2 = 80  # Minimum width of the rectangle
min_height_2 = 80  # Minimum height of the rectangle
offset_2 = 6  # Allowed error between pixels
line_position_2 = 300  # Position of the counting line
detected_2 = []
prev_centers_2 = {}  # Dictionary to store previous centers
cars_2 = 0

# Function to get center of rectangle
def get_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Function to calculate speed
def calculate_speed(curr_center, prev_center, time):
    if prev_center:
        displacement = np.sqrt((curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2)
        speed = displacement / time
        return speed
    return 0

# Function to get dominant color within a bounding box
def get_dominant_color(frame, x, y, w, h):
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    dominant_color_bin = np.argmax(hist)
    return dominant_color_bin

# Function to map bin index to color name
def bin_to_color(bin_index):
    color_ranges = {
        (0, 10): "Black",
        (10, 30): "Dark Gray",
        (30, 50): "Gray",
        (50, 70): "Light Gray",
        (70, 90): "Red",
        (90, 110): "Orange",
        (110, 130): "Yellow",
        (130, 150): "Green",
        (150, 170): "Cyan",
        (170, 190): "Blue",
        (190, 210): "Purple",
        (210, 230): "Pink",
        (230, 255): "White"
    }
    for range_, color in color_ranges.items():
        if range_[0] <= bin_index < range_[1]:
            return color
    return "Unknown"


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
    # Scale factor (pixels per meter)
    scale_factor = 10  # Example value (adjust according to your setup)

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

        # Calculate speed
        speed = calculate_speed(center, prev_centers_2.get(i), time)
        sl=[20]
        prev_centers_2.setdefault(i, {})
        if prev_centers_2[i]:
            displacement_px = np.sqrt((center[0] - prev_centers_2[i][0]) ** 2 + (center[1] - prev_centers_2[i][1]) ** 2)
            speed_pxh = displacement_px / time * 3600  # Convert px/s to px/h
            speed_kmh = speed_pxh / scale_factor * 0.001  # Convert px/h to km/h
            sl.append(speed_kmh)
        spd=sum(sl)/len(sl)
        cv2.putText(frame1, f"Speed: {str(spd)[:2]} km/h", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        prev_centers_2[i] = center

        # Get and display dominant color
        dominant_color = get_dominant_color(frame1, x, y, w, h)
        #print(dominant_color)
        color_name = bin_to_color(dominant_color)
        cv2.putText(frame1, f"Color: {color_name}",(x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1)

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
