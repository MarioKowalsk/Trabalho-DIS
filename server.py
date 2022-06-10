import socket
import threading
import concurrent.futures
import time
import numpy as np
import cv2
import scipy.linalg.blas as bla

HOST = "127.0.0.1"
PORT = 65432
H2 = np.load("H-2.npy")
H = np.load("H.npy")

def handle(conn, addr, executor):
    g = []

    name = conn.recv(1024).decode('utf-8')

    opt = conn.recv(4).decode('utf-8')
    print(opt)

    msg = conn.recv(800000)

    g = np.frombuffer(msg, dtype=np.float32)

    filename = name + "-" + str(time.time())
    np.save("./signals/" + filename, g) 

    
    if int(opt) == 1:
        executor.submit(CGNE, filename)
    elif int(opt) == 2:
        executor.submit(CGNR, filename)

    conn.close()

def CGNE(filename):
    start_time = time.time()
    g = np.load("./signals/" + filename + ".npy")

    if len(g) > 40000:
        length = 60
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

        f = np.reshape(f, (length, length))

        f = cv2.normalize(f, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        f = cv2.flip(f, 1)
        f = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        cv2.imwrite("./img/" + filename + "-CGNE.png", f)

        print("--- Completed in %s seconds ---" % (time.time() - start_time))

    else:
        length = 30
        start_time = time.time()


        f = np.zeros((pow(length, 2),), dtype=np.float32)
        r = g - bla.sgemv(1, H2, f)
        p = bla.sgemv(1, H2, r, trans=1)
        error = 1

        while error > 0.0001:
            alpha = bla.sdot(np.transpose(r), r)/ bla.sdot(np.transpose(p), p)
            fplus = bla.saxpy(p, f, a=alpha)
            rplus = r - bla.sgemv(alpha, H2, p)
            beta = bla.sdot(np.transpose(rplus), rplus)/ bla.sdot(np.transpose(r), r)
            p = bla.sscal(beta, p) + bla.sgemv(1, H2, rplus, trans=1)
            error = abs(bla.snrm2(rplus) - bla.snrm2(r))
            r = rplus
            f = fplus

        f = np.reshape(f, (length, length))

        f = cv2.normalize(f, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        f = cv2.flip(f, 1)
        f = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        cv2.imwrite("./img/" + filename + "-CGNE.png", f)

        print("--- Completed in %s seconds ---" % (time.time() - start_time))

def CGNR(filename):
   
    start_time = time.time()
    g = np.load("./signals/" + filename + ".npy")

    if len(H) > 40000:
        length = 60
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

        f = np.reshape(f, (length, length))

        f = cv2.normalize(f, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        f = cv2.flip(f, 1)
        f = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        cv2.imwrite("./img/" + filename + "-CGNR.png", f)

        print("--- Completed in %s seconds ---" % (time.time() - start_time))

    else:
        length = 30
        start_time = time.time()

        f = np.zeros((pow(length, 2),), dtype=np.float32)
        r = g - bla.sgemv(1, H2, f)
        z = bla.sgemv(1, H2, r, trans=1)
        p = z
        error = 1

        while error > 0.0001:
            w = bla.sgemv(1, H2, p)
            alpha = pow(bla.snrm2(z), 2)/ pow(bla.snrm2(w), 2)
            fplus = bla.saxpy(p, f, a=alpha)
            rplus = r - bla.sscal(alpha, w)
            zplus = bla.sgemv(1, H2, rplus, trans=1)
            beta = pow(bla.snrm2(zplus), 2)/ pow(bla.snrm2(z), 2)
            p = bla.saxpy(p, zplus, a=beta)
            error = abs(bla.snrm2(rplus) - bla.snrm2(r))
            r = rplus
            f = fplus
            z = zplus

        f = np.reshape(f, (length, length))

        f = cv2.normalize(f, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        f = cv2.flip(f, 1)
        f = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        cv2.imwrite("./img/" + filename + "-CGNR.png", f)

        print("--- Completed in %s seconds ---" % (time.time() - start_time))


def main():
    s =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

    while True:
        conn, addr = s.accept()
        thread = threading.Thread(target=handle, args=(conn, addr, executor))
        thread.start()

if __name__ == "__main__":
    main()