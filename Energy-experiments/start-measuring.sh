#!/bin/bash

while true
do
    # We need to run the python code that will open a socket conn e query the W consumption from MW100
    # MW1000 IP is 192.168.0.1 and GENE port is 34318
    # python code is responsible for half a second wait
    #date +"%y/%m/%d %T"
    #python3 measure-reader.py 192.168.0.10 34318
    python3 measure-reader.py 192.168.0.10 34318 >> Wmeasure.log
 
done

