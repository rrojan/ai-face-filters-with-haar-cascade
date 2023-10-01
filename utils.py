import cv2


def get_filter_path(filter_name):
    return f'./filters/{filter_name}-alpha.png'


def read_image(filter_path):
    sunglasses = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)

    # Resize the sunglasses to half of its original height
    sunglasses_height = int(sunglasses.shape[0] / 2)
    sunglasses_resized = cv2.resize(
        sunglasses, (sunglasses.shape[1], sunglasses_height))
    return sunglasses, sunglasses_height, sunglasses_resized


def switch_filter(current_filter, ALL_FILTERS):
    idx = ALL_FILTERS.index(filter)
    filter = ALL_FILTERS[0 if idx == len(ALL_FILTERS) - 1 else idx + 1]
