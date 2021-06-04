#!/usr/bin/env python3
from PIL import Image
import numpy as np

def g(im) :
    w, h = im.width, im.height
    pixel = im.load()
    while w//2 > 30 and h//2 > 30 :
        w = w//2
        h = h//2
        data = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(w) :
            for j in range(h) :
                r0, g0, b0 = pixel[2*i, 2*j]
                r1, g1, b1 = pixel[2*i, 2*j+1]
                r2, g2, b2 = pixel[2*i+1, 2*j]
                r3, g3, b3 = pixel[2*i+1, 2*j+1]
                r = (r0+r1+r2+r3) // 4
                g = (g0+g1+g2+g3) // 4
                b = (b0+b1+b2+b3) // 4
                data[j, i] = (r, g, b)
        newImg = Image.fromarray(data, 'RGB')
        print(newImg.size)
        pixel = newImg.load()
        newImg.show()

im = Image.open("images/arch-Ap.jpg")
print(im.size)
im.show()
g(im)
