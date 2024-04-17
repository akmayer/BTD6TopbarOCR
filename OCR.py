import pytesseract
import cv2
import imutils, time
from PIL import Image
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:your/tesseract.exe/path'

def imshow(cv2Img):
    rgb_image = cv2.cvtColor(cv2Img, cv2.COLOR_BGR2RGB)

    plt.imshow(rgb_image)
    plt.axis('off')  # Turn off axis labels
    plt.show()

def preprocess(cv2Img):
    gray = cv2.cvtColor(cv2Img, cv2.COLOR_BGR2GRAY)
    blurImg = cv2.blur(gray, (2, 2))  
    thresh = cv2.threshold(blurImg, 250, 255, cv2.THRESH_BINARY_INV)[1]
    return thresh

def extractText(npImg, rgb=True, display=False):
    if rgb:
        bgrImg = cv2.cvtColor(npImg, cv2.COLOR_RGB2BGR)
    else:
        bgrImg = npImg
    imgProc = preprocess(bgrImg)
    if display:
        imshow(imgProc)
    data = pytesseract.image_to_string(imgProc, lang='eng',config='--psm 6')
    return data