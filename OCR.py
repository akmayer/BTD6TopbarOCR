import cv2
import imutils, time
from PIL import Image
import matplotlib.pyplot as plt

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
