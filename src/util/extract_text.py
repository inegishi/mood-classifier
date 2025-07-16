import pytesseract as pyt
import cv2 as cv

pyt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_raw_text(file_path):
    ignore = ["Messages", "Details"]

    img = cv.imread(file_path)
    text = pyt.image_to_string(img)
    
    messages = [
        msg.replace("\n", " ").strip()
        for msg in text.split("\n\n")
        if msg.strip() and len(msg) >= 3 and msg.strip() not in ignore
    ]

    return text

def get_bounding_box(file_path):
    img = cv.imread(file_path,cv.IMREAD_GRAYSCALE)
    d = pyt.image_to_data(img, output_type=pyt.Output.DICT)
    print(d)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('img', img)
    cv.waitKey(0)
    return 



data_path = "dataset/test4.png"
text = extract_raw_text(data_path)
for i in text:
    print(i)