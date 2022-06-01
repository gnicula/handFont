from re import I
import constants
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

    # image = cv2.resize(image, dsize=(0, 0), fx=0.4, fy=0.4,
    #                    interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret2, th3 = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY)
    # blur = cv2.GaussianBlur(image, (3, 3), 0)
    # ret3, th3 = cv2.threshold(blur, threshold, 255,
    #                           cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), dtype=np.uint8)
    # horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # eroded = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, horizontalStructure)
    #eroded = cv2.dilate(eroded, kernel)
    th3 = cv2.bitwise_not(th3)
    # cv2.imshow("title", th3)
    # cv2.waitKey(0)
    return th3


def findContours(image):
    numCont, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) * 255
    #drawContours(image_copy, numCont, "FoundContours")

    return numCont


def getGlyphContours(contours):
    def contourGT(a):
        assert len(a.shape) and a.shape[1] == 1
        pointsA = np.squeeze(a)
        return np.max(pointsA, axis=0)[0] - np.min(pointsA, axis=0)[0]

    return sorted(contours, key=contourGT, reverse=True)[:80]


def cropGlyph(image, contour, uniVal):
    # shrink the contour so edges arent in png
    # cv2.crop(contour)
    # cv2.imsave(save with name as uniVal)
    print("contour: ", contour)
    bbox = cv2.boundingRect(contour)
    # drawContours(image, [contour], "test contour")
    print("bbox: ", bbox)
    print(image.shape)
    img = cv2.imwrite(filename=uniVal + ".png",
                      img=image[bbox[1]:bbox[3]+bbox[1], bbox[0]: bbox[2]+bbox[0]])
    return img


def createPng(image, contours):
    # sort contours first by x then by y
    # for each contour call cropGlyph()
    # need to get the unicovde from x, y center of the contour
    # and a st
    ind_to_point = []
    for i, cont in enumerate(contours):
        min_xVal_cont = 10000
        min_yVal_cont = 10000
        for point in cont:
            if point[0][0] < min_xVal_cont:
                min_xVal_cont = point[0][0]
            
            if point[0][1] < min_yVal_cont:
                min_yVal_cont = point[0][1]
        
        ind_to_point.append((min_xVal_cont, min_yVal_cont, i))
    print(ind_to_point)

    ind_to_point.sort(key=lambda x: x[0])
    ind_to_point.sort(key=lambda x: x[1])
    print(ind_to_point, ind_to_point[0][2])
    print(contours[ind_to_point[0][2]])
    cropGlyph(image, contours[ind_to_point[0][2]], "41")
    cropGlyph(image, contours[ind_to_point[1][2]], "42")
  

def createSvg(dir):
    # call potrace() to convert pngs to SVG's with the same name
    # Need subprocess function from python to call potrace as an exe
    return


if __name__ == "__main__":
    image = cv2.imread("myFont.png")
    image = cv2.resize(image, dsize=(0, 0), fx=0.4, fy=0.4,
                       interpolation=cv2.INTER_AREA)
    pimage = preProcess(image)
    listCont = findContours(pimage)
    pimageRGB = cv2.cvtColor(pimage, cv2.COLOR_GRAY2RGB) * 255
    gottenContours = getGlyphContours(listCont)
    # print(len(gottenContours))
    drawContours(pimageRGB, [gottenContours[1]], "Sorted Contours")
    # print(len(gottenContours))
    createPng(image, gottenContours)
    # cropGlyph(cv2.bitwise_not(pimage), gottenContours[0], "41") 
