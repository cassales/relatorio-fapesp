import sys
import socket
import time

if len(sys.argv) < 3:
    exit('usage: python hostname port')
#collect the arguments
hostname = sys.argv[1]
port     = int(sys.argv[2])

def netcat(hn, p, content):
    #init connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((hn, p))

    can_send = False
    # MW100 sends E0 as a first response. Aftr reading it we can send our query
    data = sock.recv(1024)
    if 'E0' in data.decode():
        can_send = True
    # once we can send our query we send it
    if can_send:
        sock.sendall(content)
        # wait a second and get the answer
        time.sleep(1)
        #get result from socket
        data = sock.recv(1024)
        # decode and split on \n
        string = data.decode().split('\n')
        # split and get the measure we are interested
        measure = string[3].split('W')[1].strip()
        # get timestamp
        sdatetime = time.strftime('%d/%m/%y,%X')
        # print
        print(f"{sdatetime},{measure}")
    # shutdown the socket
    sock.shutdown(socket.SHUT_WR)
    sock.close()

content = 'FD0,09,09\n'

netcat(hostname, port, content.encode())
