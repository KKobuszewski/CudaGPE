

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim
import scipy.optimize as o
import scipy.special as s
from scipy.fftpack import *
import glob

files = glob.glob("./data/*.bin")

print(files[0])

for f in files:
  data = np.memmap(f,dtype=np.complex128)
  #data = np.reshape(data,(-1,100))


  #x = np.linspace(-0.5,0.5,data.shape[0])
  #columns = np.arange(data.shape[1])

  print(data.shape)

  N=data.shape[0]/3

  plt.plot(data[0:N].real,label="Re fft forward")
  plt.plot(data[0:N].imag,label="Im  fft forward")
  plt.plot(data[N:2*N].real,label="Re fft in place")
  plt.plot(data[N:2*N].imag,label="Im  fft in place")
  plt.grid(True)
  plt.legend()
  plt.ylabel('transform')
  plt.xlabel('index')
  plt.xlim([0,N])
  plt.title(f)
  
  if 'fftw' in f:
    filename = './data/fftw_FFT_forward_N%d'%N
  else:
    filename = './data/cufft_FFT_forward_N%d'%N
  plt.savefig(filename)

  plt.plot(data[2*N:3*N].real,label="Re fft backward")
  plt.plot(data[2*N:3*N].imag,label="Im  fft backward")
  plt.grid(True)
  plt.legend()
  
  if 'fftw' in f:
    filename = './data/fftw_FFT_backward_N%d'%N
  else:
    filename = './data/cufft_FFT_backward_N%d'%N
  plt.savefig(filename)