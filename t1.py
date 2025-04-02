import cv2 as cv

def f():
    img = cv.imread('./res/img/blank2.png')
    cv.imshow('test', img)
    cv.waitKey(0)
    cv.destroyWindow('test')

