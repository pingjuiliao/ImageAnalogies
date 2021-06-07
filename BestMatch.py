#!/usr/bin/env python3
from annoy import AnnoyIndex
N_CHANNELS = 3
K = 0.5         # Cohereance parameter


def getFeatureVector(X, l, N, midX, midY) :
    W, H  = len(X), len(X[0])
    halfN = N // 2
    baseX, baseY = midX - halfN, midY - halfN

    vec = [ 0 for _ in range(N*N*N_CHANNELS) ]
    if l < 0 :
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
    qx, qy = q
    B_vec  = getFeatureVector(B[l], l, 5, qx, qy)
    Bp_vec = getFeatureVector(Bp[l], l, 5, qx, qy)
    BBp_vec = B_vec + Bp_vec

    W_A, H_A = len(A[l]), len(A[l][0])
    W_B, H_B = len(B[l]), len(B[l][0])

    currMin = float('Inf')
    cohX, cohY = -1, -1
    left, right = qx - 5//2, qx + 5//2
    bot, top    = qy - 5//2, qy + 5//2
    px, py = -1, -1
    searchEnd = False
    # N(r)
    for qx_neighbor in range(left, right+1) :
        for qy_neighbor in range(bot, top+1) :
            if qx_neighbor < 0 or qy_neighbor < 0 or \
                    W_B <= qx_neighbor or H_B <= qy_neighbor :
                continue
            if qx == qx_neighbor and qy == qy_neighbor :
                searchEnd = True
                break

            px, py = s[qx_neighbor][qy_neighbor]
            px_neighbor = px + (qx-qx_neighbor)
            py_neighbor = py + (qy-qy_neighbor)

            if px_neighbor < 0 or py_neighbor < 0 or \
                    W_A <= px_neighbor or H_A <= py_neighbor :
                continue
            p = (px_neighbor, py_neighbor)
            A_vec  = getFeatureVector(A[l], l, 5, px_neighbor, py_neighbor)
            Ap_vec = getFeatureVector(Ap[l], l, 5, px_neighbor, py_neighbor)
            AAp_vec= A_vec + Ap_vec

            dist = sum([(a-b)**2 for a, b in zip(AAp_vec, BBp_vec)])
            if dist < currMin :
                currMin = dist
                cohX, cohY = p
        if searchEnd :
            break

    if px == -1 or py == -1 :
        return -1, -1

    return cohX, cohY


def bestApproximateMatchAnn(A, B, l, q) :
    qx, qy = q
    B_vec  = getFeatureVector(B[l], l, 5, qx, qy)
    B_vec += getFeatureVector(B[l-1], l-1, 3, qx, qy)
    w, H = len(A[l]), len(A[l][0])
    q = AnnoyIndex(len(B_vec), 'euclidean')
    q.load('A{}.ann'.format(l))
    idx = q.get_nns_by_vector(B_vec, 1)
    pw, ph = idx[0]// H, idx[0] % H
    return pw, ph


def initSearchAnn(A, L) :
    dimension = 5 * 5 * N_CHANNELS + 3 * 3 * N_CHANNELS
    for l in range(L) :
        t = AnnoyIndex(dimension, 'euclidean')
        W, H = len(A[l]), len(A[l][0])
        for w in range(W) :
            for h in range(H) :
                idx = w*H + h
                v  = getFeatureVector(A[l], l, 5, w, h)
                v += getFeatureVector(A[l-1], l-1, 3, w//2, h//2)
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
    # Feature Approximate
    Fapp  = getFeatureVector(A[l], l, 5, app_x, app_y)
    Fapp += getFeatureVector(Ap[l], l, 5, app_x, app_y)

    # Feature Cohereance
    Fcoh  = getFeatureVector(A[l], l, 5, coh_x, coh_y)
    Fcoh += getFeatureVector(Ap[l], l, 5, coh_x, coh_y)

    # Feature B
    Fb    = getFeatureVector(B[l], l, 5, qx, qy)
    Fb   += getFeatureVector(Bp[l], l, 5, qx, qy)

    Dapp = sum([(a-b)**2 for a, b in zip(Fapp, Fb)])
    Dcoh = sum([(a-b)**2 for a, b in zip(Fcoh, Fb)])

    if Dcoh <= Dapp*(1 + 2**(l-L)) * K :
        return coh_x, coh_y
    return app_x, app_y

""" THESE TAKES HOURS

A_Feature = None # should be initialized
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
