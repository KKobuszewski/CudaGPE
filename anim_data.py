#from __future__ import divsion
#from __future__ import printing

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim
import scipy.optimize as o
import scipy.special as s
from scipy.fftpack import *


data = np.memmap("./data/wf_r_Nz1024_dt0.010000_tsteps_100.bin",dtype=np.complex128)
data = np.reshape(data,(-1,100))


x = np.linspace(-0.5,0.5,data.shape[0])
columns = np.arange(data.shape[1])

print(data.shape)

fig = plt.figure()
ax1 = fig.add_subplot(111,xlim=(-0.5,0.5),ylim=(-1.1,1.1))
psi_re, = ax1.plot([],[],label=r'$Re$ $\psi$')
psi_im, = ax1.plot([],[],label=r'$Im$ $\psi$')
#ax1.plot(x,np.abs(data[:,column]),label=r'$|\psi|$')
ax1.grid(True)
ax1.legend()
#fig.show()




def init():
    
    psi_re.set_data([],[])
    psi_im.set_data([],[])
    
    return (psi_re,psi_im)

def animate(i):
    column = columns[i]
    print(i,column)
    
    psi_re.set_data(x,data[:,column].real)
    psi_im.set_data(x,data[:,column].imag)
    
    return (psi_re,psi_im)

# specify time steps and duration
frames = data.shape[1]
print("frames:",frames)

anim = anim.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=200, blit=True)
plt.show()

