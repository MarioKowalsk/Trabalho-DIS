from distutils.log import error
import time
import numpy as np
import pandas as pd
import scipy.linalg.blas as bla
import cv2
import math

modelo = "H-2.csv"
sinal = "G-1.csv"

def CGNE():
    start_time = time.time()

    H = np.load("H.npy")
    n = 64
    if len(H) > 40000:
        s = 794
        length = 60
    else:
        s = 436
        length = 30
    g = ganhoSinal(s, n)

    print("Levou %s segundos" % (time.time() - start_time))
    start_time = time.time()


    f = np.zeros((pow(length, 2),), dtype=np.float32)
    r = g - bla.sgemv(1, H, f)
    p = bla.sgemv(1, H, r, trans=1)
    error = 1

    while error > 0.0001:
        alpha = bla.sdot(np.transpose(r), r)/ bla.sdot(np.transpose(p), p)
        fplus = bla.saxpy(p, f, a=alpha)
        rplus = r - bla.sgemv(alpha, H, p)
        beta = bla.sdot(np.transpose(rplus), rplus)/ bla.sdot(np.transpose(r), r)
        p = bla.sscal(beta, p) + bla.sgemv(1, H, rplus, trans=1)
        error = abs(bla.snrm2(rplus) - bla.snrm2(r))
        r = rplus
        f = fplus
        print('Loop, ' + str(error) )

    f = np.reshape(f, (length, length))

    filename = str(time.time()) + ".png"
    f = cv2.normalize(f, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    f = cv2.flip(f, 1)
    f = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    cv2.imwrite("./img/" + filename, f)

    print("--- Completed in %s seconds ---" % (time.time() - start_time))

def CGNR():
    start_time = time.time()

    H = pd.read_csv(modelo, dtype=np.float32, header=None)
    n = 64
    if len(H) > 40000:
        s = 794
        length = 60
    else:
        s = 436
        length = 30
    g = ganhoSinal(s, n)

    print("Levou %s segundos" % (time.time() - start_time))
    start_time = time.time()

    f = np.zeros((pow(length, 2),), dtype=np.float32)
    r = g - bla.sgemv(1, H, f)
    z = bla.sgemv(1, H, r, trans=1)
    p = z
    error = 1

    while error > 0.0001:
        w = bla.sgemv(1, H, p)
        alpha = pow(bla.snrm2(z), 2)/ pow(bla.snrm2(w), 2)
        fplus = bla.saxpy(p, f, a=alpha)
        rplus = r - bla.sscal(alpha, w)
        zplus = bla.sgemv(1, H, rplus, trans=1)
        beta = pow(bla.snrm2(zplus), 2)/ pow(bla.snrm2(z), 2)
        p = bla.saxpy(p, zplus, a=beta)
        error = abs(bla.snrm2(rplus) - bla.snrm2(r))
        r = rplus
        f = fplus
        z = zplus
        print('Loop, ' + str(error) )

    f = np.reshape(f, (length, length))

    filename = str(time.time()) + ".png"
    f = cv2.normalize(f, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    f = cv2.flip(f, 1)
    f = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    cv2.imwrite("./img/" + filename, f)

    print("--- Completed in %s seconds ---" % (time.time() - start_time))

def ganhoSinal(s, n):
    g = pd.DataFrame.to_numpy(pd.read_csv(sinal, dtype=np.float32, header=None))
    g = np.reshape(g, (n, s))
    for i in range(n):
        for j in range(s):
            gamma = 100 + 1/20*j*math.sqrt(j)
            g[i,j] = g[i,j] * gamma
    size = n*s
    g = np.reshape(g, (size, ))

    np.savetxt("./signals/teste.txt", g)
    return g

if __name__== "__main__":
    CGNE()
