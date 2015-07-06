import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim
import scipy.optimize as o
import scipy.special as s
from scipy.fftpack import *
import glob

files = glob.glob("./timing/*.bin")

print(files)

for f in files:
  data = np.memmap(f,dtype=np.float64)
  #data = np.reshape(data,(-1,100))
  print(f)
  #print(data)
  
  #print(data.shape)
  N=data.shape[0]/2
  x = np.logspace(5,4+N,num=N,base=2)
  for ii in xrange(N):
    print(x[ii],data[2*ii])
  
  plt.scatter(x,data[::2],label='Planning time')
  plt.grid(True)
  plt.legend()
  plt.ylabel('Planing time')
  plt.ylim([0,max(data[::2])*1.1])
  plt.xlabel('samples')
  plt.xlim([0.99*2**5,1.01*2**(4+N)])
  plt.xscale('log')
  plt.title(f)
  
  filename=f.replace('.bin','_plan.png')
  print(filename)
  plt.savefig(filename)
  plt.clf()
  
  for ii in xrange(N):
    print(x[ii],data[1+2*ii])
  
  plt.scatter(x,data[1::2],label="Transform time")
  plt.grid(True)
  plt.legend()
  plt.ylabel('Transform time')
  plt.ylim([0,max(data[1::2])*1.1])
  plt.xlabel('samples')
  plt.xlim([0.99*2**5,1.01*2**(4+N)])
  plt.xscale('log')
  plt.title(f)
  
  filename=f.replace('.bin','_transform.png')
  print(filename)
  plt.savefig(filename)
  plt.clf()