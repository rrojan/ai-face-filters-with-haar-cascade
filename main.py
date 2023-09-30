import cv2

# FILTER Options: sunglasses, glasses, coolglasses
FILTER = 'glasses'
ALL_FILTERS = ['glasses', 'sunglasses', 'coolglasses']
FILTER_PATH = f'./images/{FILTER}-alpha.png'
# Command I used for generating alpha channel (transparent background):
#
# convert glasses.png   -alpha on -background none -flatten glasses-alpha.png
#
# NOTE: "transparent" pngs downloaded from the internet can still pose some problems
# NOTE: so best use thi imagemagick command just in case


def read_image(filter_path):
    sunglasses = cv2.imread(FILTER_PATH, cv2.IMREAD_UNCHANGED)

    # Resize the sunglasses to half of its original height
    sunglasses_height = int(sunglasses.shape[0] / 2)
    sunglasses_resized = cv2.resize(
        sunglasses, (sunglasses.shape[1], sunglasses_height))
    return sunglasses, sunglasses_height, sunglasses_resized


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

sunglasses, sunglasses_height, sunglasses_resized = read_image(FILTER_PATH)


cap = cv2.VideoCapture(0)

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
        # TODO: Since we're working with glasses we should actually match to the eyes intead of performing full face detectin
        # TODO: and overlaying to the full fce. However eye detection is very poor
        sunglasses_resized = cv2.resize(sunglasses, (w, h))

        # Extract the region of interest
        # (this is the frame where the sunglasses will be placed
        roi = frame[y:y+h, x:x+w]

        # Create glasses mask and invert for ares
        sunglasses_alpha = sunglasses_resized[:, :, 3]
        sunglasses_area = cv2.bitwise_not(sunglasses_alpha)
        sunglasses_rgb = sunglasses_resized[:, :, :3]
        roi = cv2.bitwise_and(roi, roi, mask=sunglasses_area)

        # overlay glasses and ROI together for filter effect!
        frame[y:y+h, x:x+w] = cv2.add(roi, sunglasses_rgb)

    cv2.imshow('Haarcascade Mayajaal', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # break on "q" press
        break
    # Switch filter on "s" key press
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        idx = ALL_FILTERS.index(FILTER)
        FILTER = ALL_FILTERS[0 if idx == len(ALL_FILTERS) - 1 else idx + 1]
        sunglasses, sunglasses_height, sunglasses_resized = read_image(
            FILTER_PATH)

cap.release()
cv2.destroyAllWindows()
