import socket
import threading
import concurrent.futures
import time
import numpy as np
import cv2
import scipy.linalg.blas as bla
from datetime import datetime
import os
import glob

HOST = "127.0.0.1"
PORT = 65432


def handle(conn, addr, executor):
    g = []
    header = {
        "nome": " ",
        "alg": " ",
        "ini": " ",
        "fim": " ",
        "tam": " ",
        "itr": " ",
        "img": " "
    }

    name = conn.recv(1024).decode('utf-8')
    header["nome"] = name

    opt = conn.recv(4).decode('utf-8')

    if int(opt) == 1:
        processar(conn, header, executor)
    elif int(opt) == 2:
        recuperar(conn, header)
    elif int(opt) == 3:
        limpar()
        conn.close()

def processar(conn, header, executor):
    
    opt = conn.recv(4).decode('utf-8')

    msg = conn.recv(800000)

    g = np.frombuffer(msg, dtype=np.float32)

    filename = header["nome"] + "-" + str(time.time())
    np.save("./signals/" + filename, g) 

    
    if int(opt) == 1:
        header["alg"] = "CGNE"
        executor.submit(CGNE, filename, header)
    elif int(opt) == 2:
        header["alg"] = "CGNR"
        executor.submit(CGNR, filename, header)

    conn.close()
def recuperar(conn, header):
    with open('header.txt', 'r') as fp:
        cont = 0
        recuperated = []
        lines = fp.readlines()
        for line in lines:
            if line.find(header["nome"]) > 0:
                cont += 1
                recuperated.append(line)
        conn.send(cont.to_bytes(2, 'big'))
        if cont == 0:
            conn.send('Nenhuma imagem associada a tal usuario'.encode('utf-8'))
            return
        else:
            for i in range(cont):
                conn.send(str(recuperated[i]).encode('utf-8'))
    conn.close()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT + 1))
        s.listen()
        conn, addr = s.accept()
        msg = conn.recv(2)
        n = int.from_bytes(msg, byteorder='big')
        opts = []
        for i in range(n):
            opts.append(conn.recv(2).decode('utf-8'))
        s.listen()
        conn, addr = s.accept()
        for opt in opts:
            rec = recuperated[(int(opt) - 1)]
            length = int(rec[(rec.find("'tam'") + 7):(rec.find("'tam'") + 9)])
            conn.send(length.to_bytes(2, 'big'))
            filename = rec[(rec.find("'img':") + 8):(rec.find("'}"))]
            conn.send(np.load(filename))        
    conn.close()

def limpar():
    with open('header.txt', 'r+') as fp:
        fp.truncate(0)
        fp.close

    files = glob.glob('./img/*')
    for f in files:
        os.remove(f)
    files = glob.glob('./signals/*')
    for f in files:
        os.remove(f)
    files = glob.glob('./client/*')
    for f in files:
        os.remove(f)

def CGNE(filename, header):
    start_time = time.time()
    header["ini"] = str(datetime.today().strftime('%X %x'))
    g = np.load("./signals/" + filename + ".npy")

    if len(g) > 40000:
        length = 60
        H = np.load("H.npy")
    else:
        length = 30
        H = np.load("H-2.npy")
    header["tam"] = length
    start_time = time.time()


    f = np.zeros((pow(length, 2),), dtype=np.float32)
    r = g - bla.sgemv(1, H, f)
    p = bla.sgemv(1, H, r, trans=1)
    error = 1
    cont = 0

    while (error > 0.0001 or error == 0.0) and cont < 100:
        alpha = bla.sdot(r, r)/ bla.sdot(p, p)
        fplus = f + bla.sscal(alpha, p)
        rplus = r - bla.sgemv(alpha, H, p)
        beta = bla.sdot(rplus, rplus)/ bla.sdot(r, r)
        p =  bla.sgemv(1, H, rplus, trans=1) + bla.sscal(beta, p) 
        error = abs(bla.snrm2(rplus) - bla.snrm2(r))
        r = rplus
        f = fplus
        cont += 1

    header["itr"] = str(cont)

    f = np.reshape(f, (length, length))

    f = cv2.normalize(f, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    f = abs(f)
    f = cv2.flip(f, 1)
    f = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    np.save("./img/" + filename, f)
    header["img"] = "./img/" + filename + ".npy"
    header["fim"] = str(datetime.today().strftime('%X %x'))
    with open('header.txt', 'a') as fp:
        fp.write(str(header) + '\n')
        fp.close
    os.remove("./signals/" + filename + ".npy")
    
    del f
    del H
    del r
    del p
    del rplus
    del fplus
    
    print("--- Completed in %s seconds ---" % (time.time() - start_time))

def CGNR(filename, header):
   
    start_time = time.time()
    header["ini"] = str(datetime.today().strftime('%X %x'))
    g = np.load("./signals/" + filename + ".npy")

    if len(g) > 40000:
        length = 60
        H = np.load("H.npy")
    else:
        length = 30
        H = np.load("H-2.npy")

    header["tam"] = length
    f = np.zeros((pow(length, 2),), dtype=np.float32)
    r = g - bla.sgemv(1, H, f)
    z = bla.sgemv(1, H, r, trans=1)
    p = z
    error = 1
    cont = 0

    while (error > 0.0001 or error == 0.0) and cont < 100:
        w = bla.sgemv(1, H, p)
        alpha = pow(bla.snrm2(z), 2)/ pow(bla.snrm2(w), 2)
        fplus = f + bla.sscal(alpha, p)
        rplus = r - bla.sscal(alpha, w)
        zplus = bla.sgemv(1, H, rplus, trans=1)
        beta = pow(bla.snrm2(zplus), 2)/ pow(bla.snrm2(z), 2)
        p = zplus + bla.sscal(beta, p)
        error = abs(bla.snrm2(rplus) - bla.snrm2(r))
        r = rplus
        f = fplus
        z = zplus
        cont += 1

    header["itr"] = str(cont)

    f = np.reshape(f, (length, length))

    f = cv2.normalize(f, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    f = cv2.flip(f, 1)
    f = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    np.save("./img/" + filename, f)
    header["img"] = "./img/" + filename + ".npy"
    header["fim"] = str(datetime.today().strftime('%X %x'))
    with open('header.txt', 'a') as fp:
        fp.write(str(header) + '\n')
        fp.close
    os.remove("./signals/" + filename + ".npy")

    del f
    del H
    del r
    del p
    del rplus
    del fplus
    del z
    del zplus
    del w

    print("--- Completed in %s seconds ---" % (time.time() - start_time))

def main():
    s =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

    while True:
        conn, addr = s.accept()
        thread = threading.Thread(target=handle, args=(conn, addr, executor))
        thread.start()

if __name__ == "__main__":
    main()