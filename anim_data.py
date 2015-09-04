#from __future__ import divsion
#from __future__ import printing

import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim
import scipy.optimize as o
import scipy.special as s
from scipy.fftpack import *

Nx = 2048
data = np.memmap(glob.glob("./wf_frames*.bin")[0],dtype=np.complex128)
print(data.shape)
print('number of frames in file:',data.shape[0]/Nx)


frames_num = data.shape[0]/Nx -1
x = np.linspace(-0.5,0.5-1./Nx,Nx)
t = np.linspace(0,frames_num,frames_num+1)
X, T = np.meshgrid(x,t)
data = np.reshape(data,X.shape)

print(data.shape)
print(data)

fig = plt.figure()
ax1 = fig.add_subplot(111,xlim=(-0.5,0.5),ylim=(-1.1,1.1))
#psi_re, = ax1.plot([],[],label=r'$Re$ $\psi$')
#psi_im, = ax1.plot([],[],label=r'$Im$ $\psi$')
psi2,   = ax1.plot([],[],label=r'$|\psi|$')
phase,  = ax1.plot([],[],label=r'$arg \psi$')
ax1.grid(True)
ax1.legend()
#fig.show()




def init():
    
    #psi_re.set_data([],[])
    #psi_im.set_data([],[])
    psi2.set_data([],[])
    phase.set_data([],[])
    
    #return (psi_re,psi_im)
    return (psi2,phase)

def animate(i):
    print('frame',i)
    
    #psi_re.set_data(x,data[i,:].real)
    #psi_im.set_data(x,data[i,:].imag)
    psi2.set_data(x,np.abs(data[i,:]))
    phase.set_data(x,np.angle(data[i,:]))
    
    #return (psi_re,psi_im)
    return (psi2,phase)

# specify time steps and duration
#frames = data.shape[1]
print("frames:",frames_num)

anim = anim.FuncAnimation(fig, animate, init_func=init,
                               frames=int(frames_num), interval=200, blit=True)
plt.show()

