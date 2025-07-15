import pytesseract as pyt
import cv2 as cv

img = cv.imread("dataset/unnamed.jpg")

pyt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

text = pyt.image_to_string(img)

for line in text:
    print(line)