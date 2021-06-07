#!/usr/bin/env python3
from annoy import AnnoyIndex
N_CHANNELS = 3
A_Feature = None # should be initialized


def getFeatureVector(X, N, midX, midY) :
    W, H  = len(X), len(X[0])
    halfN = N // 2
    baseX, baseY = midX - halfN, midY - halfN

    vec = [ 0 for _ in range(N*N*N_CHANNELS) ]
    if X == None :
        return vec
    for x in range(N) :
        pixelX = baseX + x
        if pixelX < 0 or W <= pixelX :
            continue
        for y in range(N) :
            pixelY = baseY + y
            if pixelY < 0 or H <= pixelY :
                continue
            idxYIQ = x*N_CHANNELS*N + y* N_CHANNELS
            idxY, idxI, idxQ = idxYIQ, idxYIQ+1, idxYIQ+2
            vec[idxY] = X[pixelX][pixelY][0]
            vec[idxI] = X[pixelX][pixelY][1]
            vec[idxQ] = X[pixelX][pixelY][2]
    for n in vec :
        n *= N
    return vec

def bestCohereanceMatch(A, Ap, B, Bp, s, l, L, q) :
    return -1, -1


def bestApproximateMatchAnn(A, B, l, q) :
    qx, qy = q
    B_vec = getFeatureVector(B[l], 5, qx, qy)
    w, H = len(A[l]), len(A[l][0])

    q = AnnoyIndex(5*5*N_CHANNELS, 'euclidean')
    q.load('A{}.ann'.format(l))
    idx = q.get_nns_by_vector(B_vec, 1)
    pw, ph = idx[0]// H, idx[0] % H
    return pw, ph


def initSearchAnn(A, L) :
    dimension = 5 * 5 * N_CHANNELS
    for l in range(L) :
        t = AnnoyIndex(dimension, 'euclidean')
        W, H = len(A[l]), len(A[l][0])
        for w in range(W) :
            for h in range(H) :
                idx = w*H + h
                v = getFeatureVector(A[l], 5, w, h)
                t.add_item(idx, v)
        t.build(10)
        t.save("A{}.ann".format(l))


def bestMatch(A, Ap, B, Bp, s, l, L, q) :
    qx, qy = q
    app_x, app_y = bestApproximateMatchAnn(A, B, l, q)
    Bp_W, Bp_H = len(Bp[l]), len(Bp[l][0])
    if qx < 4 or qy < 4 or Bp_W-4 <= qx or Bp_H <= qy :
        return app_x, app_y
    coh_x, coh_y = bestCohereanceMatch(A, Ap, B, Bp, s, l, L, q)
    if coh_x < 0 or coh_y < 0 :
        return app_x, app_y
    return coh_x, coh_y

""" THESE TAKES HOURS
def bestApproximateMatch(B, l, q) :
    global A_Feature
    qx, qy = q
    B_vec = getFeatureVector(B[l], 5, qx, qy)
    if A_Feature is None :
        print(len(A_Feature))
        print("[ERROR] Initialization failed")
        quit()

    ## the search
    feat = A_Feature[l]
    W, H = len(feat), len(feat[0])
    argMin, currMin = (-1, -1), float("Inf")
    for w in range(W) :
        for h in range(H) :
            d = dist(B_vec, feat[w][h])
            if d < currMin :
                currMin = d
                argMin = (w, h)
   return argMin

def initSearch(A, L) :
    global A_Feature

    # constructing A_Feature
    A_Feature = []
    for l in range(L) :
        W, H = len(A[l]), len(A[l][0])
        featArr = [[None for _ in range(H)] for _ in range(W) ]
        for w in range(W) :
            for h in range(H):
                featArr[w][h] = getFeatureVector(A[l], 5, w, h)
        A_Feature.append(featArr)
    print("Initialization Done")

"""
