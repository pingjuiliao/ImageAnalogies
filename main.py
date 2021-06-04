#!/usr/bin/env python3
import os, sys
from PIL import Image
import numpy as np

#A = None
#Ap= None
#B = None

def loadImages() :
    A = Image.open("images/arch-A.jpg")
    Ap= Image.open("images/arch-Ap.jpg")
    B = Image.open("images/arch-B.jpg")

    assert( A.size == Ap.size )
    return A, Ap, B
def YIQ2RGB(arr) :
    w = len(arr)
    assert(w > 0)
    h = len(arr[0])
    assert(h > 0)
    data = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w) :
        for j in range(h) :
            Y, I, Q = arr[i][j]
            r = 1*Y + 0.956*I + 0.619*Q
            g = 1*Y - 0.272*I - 0.647*Q
            b = 1*Y - 1.106*I + 1.703*Q
            data[j, i] = (r, g, b)
    img = Image.fromarray(data, 'RGB')
    img.show()

def getYIQArray(im) :
    w, h = im.width, im.height
    pixel  = im.load()
    result = [[[0, 0, 0] for _ in range(h)] for _ in range(w) ]
    for i in range(w) :
        for j in range(h) :
            r, g, b = pixel[i, j]
            result[i][j][0] = 0.299*r + 0.587*g + 0.114*b
            result[i][j][1] = 0.596*r - 0.275*g - 0.321*b
            result[i][j][2] = 0.212*r - 0.523*g + 0.311*b
    return result
#y1y= (0.299*p1r + 0.587*p1g + 0.114*p1b)
#y1i = (0.596*p1r - 0.275*p1g - 0.321*p1b)
#y1q = (0.212*p1r - 0.523*p1g + 0.311*p1b)


def convertToYIQ(A, Ap, B) :
    yiqA = getYIQArray(A)
    yiqAp= getYIQArray(Ap)
    yiqB = getYIQArray(B)
    return yiqA, yiqAp, yiqB

def GP(arr) :

    pyramid = []
    pyramid.append(arr)

    w = len(arr)
    assert(w > 0)
    h = len(arr[0])
    assert(h > 0)
    prevArray = arr

    while w//2 > 30 and h//2 > 30 :
        w, h= w//2, h//2
        newArray = [[ [0,0,0] for j in range(h) ] for i in range(w) ]
        for i in range(w) :
            for j in range(h) :
                y0, i0, q0 = prevArray[i*2][j*2]
                y1, i1, q1 = prevArray[i*2+1][j*2]
                y2, i2, q2 = prevArray[i*2][j*2+1]
                y3, i3, q3 = prevArray[i*2+1][j*2+1]
                Y = (y0+y1+y2+y3) // 4
                I = (i0+i1+i2+i3) // 4
                Q = (q0+q1+q2+q3) // 4
                newArray[i][j][0] = Y
                newArray[i][j][1] = I
                newArray[i][j][2] = Q
        pyramid.append(newArray)

    return pyramid



def GuassianPyramid(A, Ap, B) :
    pyrA = GP(A)
    pyrAp= GP(Ap)
    pyrB = GP(B)

    """ DEBUG
    for a in pyrAp :
        YIQ2RGB(a)
    print("done")
    """
    pyrBp = []
    for a in pyrB : # a for array
        newArr = [[[0, 0, 0] for h in range(len(a[0]))] for w in range(len(a)) ]
        pyrBp.append(newArr)
    L = min(len(pyrA), len(pyrB))
    for i in range(L) :
        print(len(pyrA[i]), len(pyrA[i][0]))
        print(len(pyrAp[i]), len(pyrAp[i][0]))
        print(len(pyrB[i]), len(pyrB[i][0]))
        print(len(pyrBp[i]), len(pyrBp[i][0]))



def main() :
    A, Ap, B    = loadImages()
    yA, yAp, yB = convertToYIQ(A, Ap, B)
    GuassianPyramid(yA, yAp, yB)



if __name__ == "__main__" :
    main()
    im = Image.open("images/arch-Ap.jpg")
    p = im.load()
    for w in range(im.width) :
        for h in range(im.height) :
            r, g, b = p[w, h]
            p[w, h] = g, b, r
    im.show()