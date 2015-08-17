
# math
import numpy as np
import math

# matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# file processing
import os.path
import glob
import string
import csv


# create snapshots from frames
def process_frames(directory, evolution_type, dt,timesteps,frames, xmin, xmax, dx, Nx, omega):
  f = glob.glob(directory + 'wf_frames*.bin');
  fname = f[0]
  with open(fname) as f:
    print('content of file',fname,':')
    data = np.memmap(fname,dtype=np.complex128)
    
    print(data.shape)
    print('number of frames in file:',data.shape[0]/1024)
    
    frames_num = data.shape[0]/1024 -1
    x = np.linspace(xmin,xmax-dx,Nx)
    t = np.linspace(0,timesteps,frames_num+1)
    
    X, T = np.meshgrid(x,t)
    
    psi = np.abs(data)
    psi = np.reshape(psi,X.shape)
    print('shapes of arrays:\nX',X.shape,'\nT',T.shape,'\npsi',psi.shape)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,T,psi,rstride=8,cstride=8,alpha=0.3)
    cset = ax.contour(X, T, psi, zdir='z', offset=-1, cmap=cm.coolwarm)
    cset = ax.contour(X, T, psi, zdir='x', offset=1.1*xmin, cmap=cm.coolwarm)
    cset = ax.contour(X, T, psi, zdir='y', offset=1.1*timesteps,cmap=cm.coolwarm)
    plt.savefig(directory+'wavefunction_evolution_3dplot.pdf')
    plt.clf()
    
    
    plt.contour(X, T, psi, cmap=cm.coolwarm)
    plt.savefig(directory+'wavefunction_evolution_2dplot.pdf')
    plt.clf()
    
    figname = fname.replace('.bin','.pdf')
    with PdfPages(figname) as pdf:
        
        for ii in range(int(frames_num)):
            fig = plt.figure(dpi=500)
            plt.grid(True)
            plt.xlabel('x')
            plt.ylabel('$|\psi|^2$')
            plt.xlim([xmin,xmax])
            plt.plot(x,np.abs(data[:Nx]),label='initial $|\psi|^2$$')
            plt.plot(x,np.abs(data[Nx*ii:Nx*(ii+1)]),label='frame {}'.format(ii+1))
            plt.legend(loc='upper left')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    
    
    
    return 0

process_frames('', 'imaginary time', 1e-07, 1e7, 20, -0.5, 0.5, 1./1024., 1024, 3216.990877275948151)