#!/usr/bin/env python3
import os, sys
from PIL import Image
import numpy as np
from BestMatch import bestMatch, initSearchAnn

def loadImages(s) :

    A = Image.open("images/"+s+"/A.jpg")
    Ap= Image.open("images/"+s+"/Ap.jpg")
    B = Image.open("images/"+s+"/B.jpg")
    A.show()
    Ap.show()
    B.show()
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
        pyramid = [ newArray ] + pyramid
        prevArray = newArray

    return pyramid



def GuassianPyramid(A, Ap, B) :
    pyrA = GP(A)
    pyrAp= GP(Ap)
    pyrB = GP(B)

    # debug
    #for a in pyrAp :
    #    YIQ2RGB(a)
    #print("done")

    pyrBp = []
    for a in pyrB : # a for array
        newArr = [[[0, 0, 0] for h in range(len(a[0]))] for w in range(len(a)) ]
        pyrBp.append(newArr)

    # DEBUG
    L = min(len(pyrA), len(pyrB))
    for i in range(L) :
        assert(len(pyrA[i]) == len(pyrAp[i]))
        assert(len(pyrA[i][0]) == len(pyrAp[i][0]))
        assert(len(pyrB[i]) == len(pyrBp[i]))
        assert(len(pyrB[i][0]) == len(pyrBp[i][0]))

    return pyrA, pyrAp, pyrB, pyrBp


def main() :
    if len(sys.argv) < 2 :
        print("Usage: ./{} <image directory>", sys.argv[0])
        print("Available options: ")
        for dname in os.listdir("./images") :
            print("\t{}".format(dname))
        quit()

    imgA, imgAp, imgB    = loadImages(sys.argv[1])
    yA, yAp, yB = convertToYIQ(imgA, imgAp, imgB)

    # Guassian Pyramid
    A, Ap, B, Bp = GuassianPyramid(yA, yAp, yB)
    L    = min(len(A), len(B))

    #
    initSearchAnn(A, L)

    for l in range(L) : ## from coarse to finest
        currLayer = Bp[l]
        W, H = len(currLayer), len(currLayer[0])
        s = [[ (0, 0) for _ in range(H)] for _ in range(W) ]
        for w in range(W) :  ## scan-line order
            for h in range(H) :
                q = (w, h)
                p = bestMatch(A, Ap, B, Bp, s, l, L, q)
                pw, ph = p
                currLayer[w][h] = Ap[l][pw][ph]
                # currLayer[w][h][1] = B[l][w][h][1]
                # currLayer[w][h][2] = B[l][w][h][2]
                s[w][h] = p
        print("Layer ", l, "done!" )
        # YIQ2RGB(currLayer)

    YIQ2RGB(Bp[L-1])

if __name__ == "__main__" :
    main()
