import cv2
import numpy as np

image = cv2.imread('ocr_cin/temp/eroded.jpg')

# Perform Gaussian smoothing
blurred = cv2.GaussianBlur(image, (3, 3), 0)

# Perform Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Create a binary mask with the edges
mask = np.zeros_like(edges)
mask1 = np.zeros_like(edges)
# Draw the regions you want to keep visible on the mask using white color
cv2.rectangle(mask1, (250, 145), (430, 190), (255, 255, 255), -1)

masked_image1 = cv2.bitwise_and(image,image, mask=mask1)

cv2.imwrite('ocr_cin/temp/masked_image1.jpg', masked_image1)


cv2.rectangle(mask, (225, 190), (568, 250), (255, 255, 255), -1)
cv2.rectangle(mask, (225, 250), (560, 282), (255, 255, 255), -1)
cv2.rectangle(mask, (225, 288), (625, 320), (255, 255, 255), -1)
cv2.rectangle(mask, (225, 320), (515, 365), (255, 255, 255), -1)
cv2.rectangle(mask, (225, 365), (570, 410), (255, 255, 255), -1)

masked_image = cv2.bitwise_and(image,image, mask=mask)

cv2.imwrite('ocr_cin/temp/masked_image.jpg', masked_image)
