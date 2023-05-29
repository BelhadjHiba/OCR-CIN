import os
import pytesseract
import cv2
from PIL import Image
import arabic_reshaper
from bidi.algorithm import get_display 
import json


# Check if the JSON file exists and is not empty
if os.path.exists('ocr_cin/ocr_result.json') and os.stat('ocr_cin/ocr_result.json').st_size > 0:
    # Load existing JSON data from file
    with open('ocr_cin/ocr_result.json', 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
else:
    # Initialize an empty dictionary
    existing_data = {}


no_noise = "ocr_cin/temp/masked_image.jpg"

img = Image.open(no_noise)

config = '--psm 6 -l ara --oem 1'
lang = 'ara'
ocr_result = pytesseract.image_to_string(img, lang=lang, config=config)

# Reshape Arabic text
reshaped_text = arabic_reshaper.reshape(ocr_result)

# Convert the reshaped text to a form that can be displayed properly
display_text = get_display(reshaped_text)

# Print the final result
print(display_text)

# Split the text on newlines and count the resulting strings
num_lines = len(reshaped_text.split('\n'))
print(num_lines)


# Split the text into lines
lines = reshaped_text.split('\n')

# Create a dictionary to map key names to lines of text
key_names = ['cinNumber', 'lastName', 'firstName1', 'firstName2', 'birthDate', 'birthPlace']
result_dict = {}
for i, line in enumerate(lines):
    if i < len(key_names) - 1 :
        result_dict[key_names[i+1]] = line





#ocr cin number
no_noise = "ocr_cin/temp/masked_image1.jpg"

img = Image.open(no_noise)

config = '--psm 7 outputbase digits'
ocr_result = pytesseract.image_to_string(img, config=config)

# Convert the reshaped text to a form that can be displayed properly
display_text = get_display(ocr_result)

# Print the final result
print(display_text)

# Split the text on newlines and count the resulting strings
num_lines = len(display_text.split('\n'))
print(num_lines)
line = display_text.split('\n')


result_dict[key_names[0]] = line[0]

# Add new data to Python object
existing_data[line[0]] = result_dict

# Save the dictionary to a JSON file
with open('ocr_cin/ocr_result.json', 'w', encoding='utf-8') as f:
    json.dump(existing_data, f, ensure_ascii=False, indent=4)
