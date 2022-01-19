import cv2
import numpy as np


def drawContours(image, contours, windowName):
    image_copy = image.copy()

    if len(image_copy.shape) < 3:
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB) * 255
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1,
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow(windowName, image_copy)
    cv2.waitKey(0)


def preProcess(image, threshold=100):

    image = cv2.resize(image, dsize=(0, 0), fx=0.4, fy=0.4,
                       interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret2, th3 =  cv2.threshold(image, 160, 255, cv2.THRESH_BINARY)
    # blur = cv2.GaussianBlur(image, (3, 3), 0)
    # ret3, th3 = cv2.threshold(blur, threshold, 255,
    #                           cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), dtype=np.uint8)
    # horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # eroded = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, horizontalStructure)
    #eroded = cv2.dilate(eroded, kernel)
    th3 = cv2.bitwise_not(th3)
    cv2.imshow("title", th3)
    cv2.waitKey(0)
    return th3


def findContours(image):
    numCont, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) * 255
    drawContours(image_copy, numCont, "FoundContours")

    return numCont


def getGlyphContours(contours):
    def contourGT(a):
        assert len(a.shape) and a.shape[1] == 1
        pointsA = np.squeeze(a)
        return np.max(pointsA, axis=0)[0] - np.min(pointsA, axis=0)[0]

    return sorted(contours, key=contourGT, reverse=True)[:80]


if __name__ == "__main__":
    image = cv2.imread("myFont.png")
    pimage = preProcess(image)
    listCont = findContours(pimage)
    pimageRGB = cv2.cvtColor(pimage, cv2.COLOR_GRAY2RGB) * 255
    gottenContours = getGlyphContours(listCont)
    print(gottenContours)
    drawContours(pimageRGB, gottenContours, "Sorted Contours")
    print(len(gottenContours))
