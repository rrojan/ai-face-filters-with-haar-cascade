import cv2
from utils import get_filter_path, read_image, switch_filter
from time import time, sleep

# Set default filter as 'glasses'
filter = 'glasses'
filter_path = get_filter_path(filter)
ALL_FILTERS = ['glasses', 'sunglasses', 'coolglasses']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

sunglasses, sunglasses_height, sunglasses_resized = read_image(filter_path)

cap = cv2.VideoCapture(0)
photo_captured = False
photo_captured_time = 0

while True:
    # Capture single frame from webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Resize the sunglasses image to match the dimensions of the detected face (ROI)
        # TODO: Since we're working with glasses we should actually match to the eyes intead of performing full face detection
        # TODO: and overlaying to the full face.
        sunglasses_resized = cv2.resize(sunglasses, (w, h))

        # Extract the region of interest
        # (this is the frame where the sunglasses will be placed)
        roi = frame[y:y+h, x:x+w]

        # Create glasses mask and invert for areas
        sunglasses_alpha = sunglasses_resized[:, :, 3]
        sunglasses_area = cv2.bitwise_not(sunglasses_alpha)
        sunglasses_rgb = sunglasses_resized[:, :, :3]
        roi = cv2.bitwise_and(roi, roi, mask=sunglasses_area)

        # overlay glasses and ROI together for filter effect!
        frame[y:y+h, x:x+w] = cv2.add(roi, sunglasses_rgb)

    if photo_captured:
        current_time = time()
        if current_time - photo_captured_time <= 2:
            cv2.putText(frame, "Photo Captured", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Haarcascade Mayajaal', frame)

    # Capture a screenshot when spacebar is pressed
    key = cv2.waitKey(1)
    if key == ord(' '):
        screenshot_name = f'screenshot-{int(time())}.png'
        cv2.imwrite(screenshot_name, frame)
        print(f'Screenshot saved as {screenshot_name}')
        photo_captured = True
        photo_captured_time = time()

    # Break on "q" press
    if key == ord('q'):
        break
    # Switch filter on "s" key press
    elif key == ord('s'):
        filter = switch_filter(filter, ALL_FILTERS)
        filter_path = get_filter_path(filter)
        sunglasses, sunglasses_height, sunglasses_resized = read_image(
            filter_path)

cap.release()
cv2.destroyAllWindows()
