import cv2
from matplotlib import pyplot as plt
import numpy as np

# opening the image
image_file = "ocr_cin/temp/resized_identity_card.jpg"
img = cv2.imread(image_file)

# display the image
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

# invert the image 
inverted_image = cv2.bitwise_not(img)
cv2.imwrite("ocr_cin/temp/inverted.jpg", inverted_image)

# convert image to GrayScale
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = grayscale(inverted_image)
cv2.imwrite("ocr_cin/temp/gray.jpg", gray_image)

# Apply binary thresholding to the grayscale image
border_size = 60
min_brightness = gray_image[border_size:-border_size, border_size:-border_size].mean()
threshold_value = min_brightness + 51
print(threshold_value)
    
# Apply inverse binary thresholding
_, bw_im = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

cv2.imwrite("ocr_cin/temp/bw_image.jpg", bw_im)
display("ocr_cin/temp/bw_image.jpg")

# noise removal
def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    denoised = cv2.bilateralFilter(image,9,75,75)
    return (denoised)

# remove noise from the binary image
no_noise = noise_removal(bw_im)
cv2.imwrite("ocr_cin/temp/no_noise.jpg", no_noise)
display("ocr_cin/temp/no_noise.jpg")

# apply image sharpening
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened_image = cv2.filter2D(no_noise, -1, sharpen_kernel)
cv2.imwrite("ocr_cin/temp/sharpened.jpg", sharpened_image)
display("ocr_cin/temp/sharpened.jpg")

# apply morphological transformations for text enhancement
kernel = np.ones((2,2), np.uint8)
dilated_image = cv2.dilate(sharpened_image, kernel, iterations=2)
cv2.imwrite("ocr_cin/temp/dilated.jpg", dilated_image)
display("ocr_cin/temp/dilated.jpg")
