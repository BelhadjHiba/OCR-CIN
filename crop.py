import cv2
import numpy as np

# Load the image
img = cv2.imread('ocr_cin/img/cin1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image to create a binary image
ret, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

# Find the contours in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area (which should correspond to the identity card)
max_area = 0
max_contour = None
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

# Find the minimum area rectangle of the largest contour
rect = cv2.minAreaRect(max_contour)

# Determine the angle needed to make the rectangle upright
angle = rect[2]
if angle < -45:
    angle += 90

# Rotate the image to make the rectangle upright
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Find the bounding rectangle for the largest contour on the rotated image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
max_area = 0
max_contour = None
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour
x, y, w, h = cv2.boundingRect(max_contour)

# Crop the rotated image using the bounding rectangle
cropped_img = rotated_img[y:y+h, x:x+w]

# Save the cropped image
cv2.imwrite('ocr_cin/temp/cropped_identity_card.jpg', cropped_img)

resized_image = cv2.resize(cropped_img, (650,430))
cv2.imwrite('ocr_cin/temp/resized_identity_card.jpg', resized_image)
