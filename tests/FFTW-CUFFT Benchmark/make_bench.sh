#!/bin/bash


unset NFFTS


#echo "./bench.np2.oop.serial"
#./bench.np2.oop.serial > out/bench.np2.oop.serial
#echo "./bench.np2.oop.threads"
#./bench.np2.oop.threads > out/bench.np2.oop.threads
#echo "./bench.np2.oop.cuda"
#./bench.np2.oop.cuda > out/bench.np2.oop.cuda

#echo "./bench.p2.oop.serial"
#touch out/bench.p2.oop.serial.txt
# ./bench.p2.oop.serial > out/bench.p2.oop.serial.txt
echo "./bench.p2.oop.threads"
touch out/bench.p2.oop.threads.txt
./bench.p2.oop.threads > out/bench.p2.oop.threads.tx
echo "./bench.p2.oop.cuda"
touch out/bench.p2.oop.cuda.txt
./bench.p2.oop.cuda > out/bench.p2.oop.cuda.txt

#echo "./bench.np2.ip.serial"
#./bench.np2.ip.serial > out/bench.np2.ip.serial
#echo "./bench.np2.ip.threads"
#./bench.np2.ip.threads > out/bench.np2.ip.threads
#echo "./bench.np2.ip.cuda"
#./bench.np2.ip.cuda > out/bench.np2.ip.cuda

#echo "./bench.p2.ip.serial"
#touch out/bench.p2.ip.serial.txt
# ./bench.p2.ip.serial > out/bench.p2.ip.serial.txt
echo "./bench.p2.ip.threads"
touch out/bench.p2.ip.threads.txt
./bench.p2.ip.threads > out/bench.p2.ip.threads.txt
echo "./bench.p2.ip.cuda"
touch out/bench.p2.ip.cuda.txt
./bench.p2.ip.cuda > out/bench.p2.ip.cuda.txt

echo "./bench.p2.ip.dev.cuda"
touch out/bench.p2.ip.dev.cuda.txt
./bench.p2.ip.dev.cuda > out/bench.p2.ip.dev.cuda.txt
#echo "./bench.np2.ip.dev.cuda"
#./bench.np2.ip.dev.cuda > out/bench.np2.ip.dev.cuda
echo "./bench.p2.oop.dev.cuda"
touch out/bench.p2.oop.dev.cuda.txt
./bench.p2.oop.dev.cuda > out/bench.p2.oop.dev.cuda.txt
#echo "./bench.np2.oop.dev.cuda"
#./bench.np2.oop.dev.cuda > out/bench.np2.oop.dev.cuda

#TWOD

#echo "./bench.np2.oop.serial.2d"
#./bench.np2.oop.serial.2d > out/bench.np2.oop.serial.2d
#echo "./bench.np2.oop.threads.2d"
#./bench.np2.oop.threads.2d > out/bench.np2.oop.threads.2d
#echo "./bench.np2.oop.cuda.2d"
#./bench.np2.oop.cuda.2d > out/bench.np2.oop.cuda.2d

#echo "./bench.p2.oop.serial.2d"
#touch out/bench.p2.oop.serial.2d.txt
# ./bench.p2.oop.serial.2d > out/bench.p2.oop.serial.2d.txt
echo "./bench.p2.oop.threads.2d"
touch out/bench.p2.oop.threads.2d.txt
./bench.p2.oop.threads.2d > out/bench.p2.oop.threads.2d.txt
echo "./bench.p2.oop.cuda.2d"
touch out/bench.p2.oop.cuda.2d.txt
./bench.p2.oop.cuda.2d > out/bench.p2.oop.cuda.2d.txt

#echo "./bench.np2.ip.serial.2d"
#./bench.np2.ip.serial.2d > out/bench.np2.ip.serial.2d
#echo "./bench.np2.ip.threads.2d"
#./bench.np2.ip.threads.2d > out/bench.np2.ip.threads.2d
#echo "./bench.np2.ip.cuda.2d"
#./bench.np2.ip.cuda.2d > out/bench.np2.ip.cuda.2d

# echo "./bench.p2.ip.serial.2d"
# touch out/bench.p2.ip.serial.2d.txt
# ./bench.p2.ip.serial.2d > out/bench.p2.ip.serial.2d.txt
echo "./bench.p2.ip.threads.2d"
touch out/bench.p2.ip.threads.2d.txt
./bench.p2.ip.threads.2d > out/bench.p2.ip.threads.2d.txt
echo "./bench.p2.ip.cuda.2d"
touch out/bench.p2.ip.cuda.2d.txt
./bench.p2.ip.cuda.2d > out/bench.p2.ip.cuda.2d.txt

echo "./bench.p2.ip.dev.cuda.2d"
touch out/bench.p2.ip.dev.cuda.2d.txt
./bench.p2.ip.dev.cuda.2d > out/bench.p2.ip.dev.cuda.2d.txt
#echo "./bench.np2.ip.dev.cuda.2d"
#./bench.np2.ip.dev.cuda.2d > out/bench.np2.ip.dev.cuda.2d
echo "./bench.p2.oop.dev.cuda.2d"
touch out/bench.p2.oop.dev.cuda.2d.txt
./bench.p2.oop.dev.cuda.2d > out/bench.p2.oop.dev.cuda.2d.txt
#echo "./bench.np2.oop.dev.cuda.2d"
#./bench.np2.oop.dev.cuda.2d > out/bench.np2.oop.dev.cuda.2d

#THREED

#echo "./bench.np2.oop.serial.3d"
#./bench.np2.oop.serial.3d > out/bench.np2.oop.serial.3d
#echo "./bench.np2.oop.threads.3d"
# ./bench.np2.oop.threads.3d > out/bench.np2.oop.threads.3d
# echo "./bench.np2.oop.cuda"
# ./bench.np2.oop.cuda.3d > out/bench.np2.oop.cuda.3d

# echo "./bench.p2.oop.serial.3d"
# touch out/bench.p2.oop.serial.3d
# ./bench.p2.oop.serial.3d > out/bench.p2.oop.serial.3d
echo "./bench.p2.oop.threads.3d"
touch out/bench.p2.oop.threads.3d
./bench.p2.oop.threads.3d > out/bench.p2.oop.threads.3d
echo "./bench.p2.oop.cuda.3d"
touch out/bench.p2.oop.cuda.3d
./bench.p2.oop.cuda.3d > out/bench.p2.oop.cuda.3d

# echo "./bench.np2.ip.serial.3d"
# ./bench.np2.ip.serial.3d > out/bench.np2.ip.serial.3d
# echo "./bench.np2.ip.threads.3d"
# ./bench.np2.ip.threads.3d > out/bench.np2.ip.threads.3d
# echo "./bench.np2.ip.cuda.3d"
# ./bench.np2.ip.cuda.3d > out/bench.np2.ip.cuda.3d

# echo "./bench.p2.ip.serial.3d"
# touch out/bench.p2.ip.serial.3d
# ./bench.p2.ip.serial.3d > out/bench.p2.ip.serial.3d
echo "./bench.p2.ip.threads.3d"
touch out/bench.p2.ip.threads.3d
./bench.p2.ip.threads.3d > out/bench.p2.ip.threads.3d
echo "./bench.p2.ip.cuda.3d"
touch out/bench.p2.ip.cuda.3d
./bench.p2.ip.cuda.3d > out/bench.p2.ip.cuda.3d

echo "./bench.p2.ip.dev.cuda.3d"
touch out/bench.p2.ip.dev.cuda.3d
./bench.p2.ip.dev.cuda.3d > out/bench.p2.ip.dev.cuda.3d
# echo "./bench.np2.ip.dev.cuda.3d"
# ./bench.np2.ip.dev.cuda.3d > out/bench.np2.ip.dev.cuda.3d
echo "./bench.p2.oop.dev.cuda.3d"
touch out/bench.p2.oop.dev.cuda.3d
./bench.p2.oop.dev.cuda.3d > out/bench.p2.oop.dev.cuda.3d
# echo "./bench.np2.oop.dev.cuda.3d"
# ./bench.np2.oop.dev.cuda.3d > out/bench.np2.oop.dev.cuda.3d
