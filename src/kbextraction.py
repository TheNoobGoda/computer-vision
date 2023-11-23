import cv2
import numpy as np

def find_piano(image_path):
    # Read the input image
    original_image = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Define a mask for the color range of white keys
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

    piano_image = cv2.bitwise_and(original_image, original_image, mask=mask_white)
    cv2.imshow('Piano Image', piano_image)

    # Find contours in the white mask
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box of the combined contours
    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    # Crop the original image to the bounding box
    cropped_image = original_image[y:y+h, x:x+w]

    return cropped_image

# Example usage
image_path = 'imagepiano.jpg'
result_image = find_piano(image_path)

# Display the result
cv2.imshow('Cropped Piano Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
