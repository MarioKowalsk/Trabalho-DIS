import socket
import pandas as pd
import numpy as np
import cv2
import math

HOST = "127.0.0.1"
PORT = 65432


def main():
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.connect((HOST, PORT))

    connected = True
    msg = input("Nome do usuario: ")
    c.send(msg.encode('utf-8'))
    opt = input("Deseja:\n 1.Enviar Sinal\n 2.Baixar Imagens\n 3.Limpar pastas\n> ")
    c.send(opt.encode('utf-8'))

    if int(opt) == 1:
        enviarSinal(c)
    elif int(opt) == 2:
        recuperarImagens(msg, c)
    elif int(opt) == 3:
        c.close()


def enviarSinal(c):
    msg = input("Deseja:\n 1.CGNE\n 2.CGNR\n> ")
    c.send(msg.encode('utf-8'))

    filename = input("Nome do arquivo: ")

    g = pd.DataFrame.to_numpy(pd.read_csv(filename, dtype=np.float32, header=None))

    opt = input("Ganho de Sinal?\n1.Sim\n2.Não\n> ")
    if int(opt) == 1:
        n = 64
        if len(g) > 40000:
            s = 794
        else:
            s = 436
        g = np.reshape(g, (n, s))
        for i in range(n):
            for j in range(s):
                gamma = 100 + 1/20*j*math.sqrt(j)
                g[i,j] = g[i,j] * gamma
        size = n*s
        g = np.reshape(g, (size, ))
    c.send(g.tobytes())
    c.close()

def recuperarImagens(name, c):
    msg = c.recv(4)
    n = int.from_bytes(msg, byteorder='big')
    if n == 0:
        msg = c.recv(1024).decode('utf-8')
        print(msg)
        c.close()
        return
    else:  
        for i in range(n):
            msg = c.recv(1024).decode('utf-8')
            print(str(i) + " " + msg)
        c.close()

        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c.connect((HOST, PORT + 1))
        opts = input("Escolha as imagens a serem recuperada: ")
        opts = opts.split()
        
        c.send(len(opts).to_bytes(2, 'big'))
        for opt in opts:
            c.send(opt.encode('utf-8'))
        c.close()
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c.connect((HOST, PORT + 1))

        for i in range(len(opts)):
            length = int.from_bytes(c.recv(2), byteorder='big')
            msg = c.recv(30000)
            f = np.frombuffer(msg, dtype=np.float32)

            f = np.reshape(f, (length, length))
            cv2.imwrite("./client/" + name + "-" + str(i) + ".png", f)
            print("Imagem recebida no cliente")
        c.close()


if __name__ == "__main__":
    main()