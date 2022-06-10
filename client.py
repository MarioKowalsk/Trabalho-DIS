import socket
import pandas as pd
import numpy as np
import math

HOST = "127.0.0.1"
PORT = 65432


def main():
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.connect((HOST, PORT))

    connected = True
    msg = input("Nome do usuario: ")
    c.send(msg.encode('utf-8'))
    msg = input("Deseja:\n 1.CGNE\n 2.CGNR\n> ")
    c.send(msg.encode('utf-8'))

    filename = input("Nome do arquivo: ")

    g = pd.DataFrame.to_numpy(pd.read_csv(filename, dtype=np.float32, header=None))
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

if __name__ == "__main__":
    main()