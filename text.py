import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import image_to_string

#print(get_string('text.png'))
def text(img_data):
	return(pytesseract.image_to_string(img_data))