
# math
import numpy as np
import math

# matplotlib
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# file processing
import os.path
import glob
import string
import csv



def get_params():
  # simulation parameters definitions
  evolution_type = None
  dt = None
  timesteps = None
  frames = None
  xmin = None
  xmax = None
  dx = None
  Nx = None
  omega = None
  
  # processing simulation params
  f = glob.glob(directory + 'simulation_parametrs*');
  f = f[0]
  with open(f) as csvfile:
    print('content of file',f,':')
    
    csvreader = csv.reader(csvfile, delimiter='\t')
    for row in csvreader:
      #print('\t'.join(row))
      #print('print("',row[0],'",',row[0].replace(':',''),',type(',row[0].replace(':',''),'))')
      if (row[0] == 'evolution:'):
        evolution_type=row[1]
      if (row[0] == 'dt:'):
        dt=np.float64(row[1])
      if (row[0] == 'timesteps made:'):
        timesteps=int(row[1])
      if (row[0] == 'frames:'):
        frames=int(row[1])
      if (row[0] == 'xmin:'):
        xmin=np.float64(row[1])
      if (row[0] == 'xmax:'):
        xmax=np.float64(row[1])
      if (row[0] == 'dx:'):
        dx=np.float64(row[1])
      if (row[0] == 'Nx:'):
        Nx=int(row[1])
      if (row[0] == 'harmonic potential angular freq.:'):
        omega=np.float64(row[1])
      # TODO:
      # checking potentials!
      # make it in dict!
    
    # check it
    print('evolution:',evolution_type,type(evolution_type))
    print('dt:',dt,type(dt))
    print("timesteps:", timesteps ,type( timesteps ))
    print("frames:", frames ,type( frames ))
    print("Nx:", Nx ,type( Nx ))
    print("xmin:", xmin ,type( xmin ))
    print("xmax:", xmax ,type( xmax ))
    print("dx:", dx,type(dx))
    print("harmonic potential angular freq.:", omega,type(omega))
    print()
    
    return (evolution_type, dt,timesteps,frames, xmin, xmax, dx, Nx, omega)
# end of processing simulation params


def process_statistics(directory, evolution_type, dt,timesteps,frames, xmin, xmax, dx, Nx, omega):
  # process statistics
  fname = glob.glob(directory + 'stat*.txt')[0]
  with open(fname) as f:
      print('STATISTICS\nfile:',f,':')
      
      data = np.loadtxt(f, skiprows=5)
      t = data[:,0]
      n = data[:,1]
      mu = data[:,2]
      T = data[:,3]
      Vext = data[:,4]
      Vcon = data[:,5]
      Vdip = data[:,6]

      virial = 2*T - 2*Vext
      Etot = T + Vext
      """
      print('norm change',np.max(n) - np.min(n),'per',np.max(t),'steps (',(np.max(n) - np.min(n))/np.max(t),')')
      print('kinetic energy  mean:',np.mean(T),'  min:',np.min(T),'  std. dev:',np.std(T))
      print('ext. pot. energy  mean:',np.mean(Vext),'  min:',np.min(Vext),'  std. dev:',np.std(Vext))
      print('total energy  mean:',np.mean(Etot),'  min:',np.min(Etot),'  std. dev:',np.std(Etot))
      print('virial  mean:',np.mean(virial),'  min:',np.min(virial),'  std. dev:',np.std(virial))
      """
      stats_str = ''
      if evolution_type == 'real time':
        stats_str = (
              'evolution: {}\n'.format(evolution_type),
              'Nx: {}\n'.format(Nx),
              'x in [{},{}]\t'.format(xmin,xmax),
              'dx: {}\n'.format(dx),
              'omega: {}\n'.format(omega),
              'statistics:',
              'norm change: {} per {} steps ({} per step)\n\n'.format(np.max(n) - np.min(n),np.max(t),(np.max(n) - np.min(n))/np.max(t)),
              'kinetic energy\nmean: {}   min:{}   std. dev.:{}\n\n'.format(np.mean(T),np.min(T),np.std(T)),
              'ext. pot. energy\nmean: {}   min:{}   std. dev.:{}\n\n'.format(np.mean(Vext),np.min(Vext),np.std(Vext)),
              'total energy\nmean: {}   min:{}   std. dev.:{}\n\n'.format(np.mean(Etot),np.min(Etot),np.std(Etot)),
              '|virial|\nmean: {}   max:{}   std. dev.:{}\n\n'.format(np.mean(virial),np.max(np.abs(virial)),np.std(virial))
              )
      if evolution_type == 'imaginary time':
        stats_str = (
              'evolution: {}\n'.format(evolution_type),
              'Nx: {}\n'.format(Nx),
              'x in [{},{}]\t'.format(xmin,xmax),
              'dx: {}\n'.format(dx),
              'omega: {}\n'.format(omega),
              'statistics:',
              'norm change: {} per {} steps ({} per step)\n\n'.format(np.max(n) - np.min(n),np.max(t),(np.max(n) - np.min(n))/np.max(t)),
              'kinetic energy\nmin:{}\n\n'.format(np.min(T)),
              'ext. pot. energy\nmin:{}\n\n'.format(np.min(Vext)),
              'total energy\n\nmin:{}\n\n'.format(np.min(Etot)),
              '|virial|\nmin:{}\n\n'.format(np.min(np.abs(virial))),
              'chemical potential\nmin:{}\n\n'.format(np.min(mu))
              )
      stats_str = ''.join(stats_str)
      
      figname = fname.replace('.txt','_dt%.2e.pdf'%dt)
      with PdfPages(figname) as pdf:
          #
          # text page
          #plt.rc('text',usetex=True)
          fig = plt.figure(figsize=(14,10),dpi=500)
          #plt.title('')
          plt.axis('off')
          plt.text(0,0,stats_str)
          pdf.savefig(fig)
          plt.close()
          
          # energy
          mpl.rcParams["axes.formatter.useoffset"] = False
          plt.margins(1)
          fig = plt.figure(figsize=(8,8),dpi=1000)
          plt.grid(True)
          if evolution_type == 'imaginary time':
            plt.xscale('log')
            plt.yscale('log')
          plt.title('mean values of different energy operators')
          plt.xlabel('timesteps')
          plt.ylabel('energy')
          plt.ylim([0.99*min( np.min(T),np.min(Vext) ), 1.01*max( np.max(T),np.max(Vext) )])
          plt.plot(t,T,label='T')
          plt.plot(t,Vext,label=r'$V_{ext}$')
          #plt.plot(t,Vcon,label=r'$V_{con}$')
          #plt.plot(t,Vdip,label=r'$V_{dip}$')
          plt.plot(t,(T+Vext+Vcon+Vdip)/2,label='srednia')
          plt.legend()
          plt.yticks(rotation='vertical')
          #ax = plt.gca()
          #ax.get_yaxis().get_major_formatter().set_useOffset(False)
          plt.tight_layout()
          plt.subplots_adjust()
          pdf.savefig(fig)
          plt.close()
          
          # chemical potential
          if evolution_type == 'imaginary time':
            fig = plt.figure(figsize=(8,8),dpi=1000)
            plt.grid(True)
            plt.xscale('log')
            plt.yscale('log')
            plt.ylim([np.min(mu)*0.99,np.max(mu)*1.01])
            plt.title('chemical potential')
            plt.plot(t,mu,label=r'$\mu$')
            plt.legend()
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
          
          #virial 
          #plt.rc('text',usetex=True)
          fig = plt.figure(figsize=(8,8),dpi=1000)
          plt.grid(True)
          if evolution_type == 'imaginary time':
            plt.xscale('log')
            plt.yscale('log')
          plt.title('virial and changes of total energy')
          plt.ylim([0.99*min( np.min(virial),np.min(Etot - np.min(Etot)) ), 1.01*max( np.max(virial),np.max(Etot - np.min(Etot)) )])
          plt.plot(t,virial,label='Virial')
          plt.plot(t,Etot - np.min(Etot),label=r'$E_{tot} - E_{min}$')
          plt.legend()
          plt.tight_layout()
          pdf.savefig(fig)
          plt.close()
          
          if evolution_type == 'real time':
            fig = plt.figure(figsize=(8,8),dpi=1000)
            plt.grid(True)
            plt.title('changes of norm of wavefunction')
            plt.plot(t,n,label=r'norm')
            plt.legend()
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
          
          # here add some more PdfPages
          #
# end of statistics processing


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



dirs = glob.glob('./data*/')
print('directories:')
for d in dirs:
    print(d)
print('__________________________________________________________________________________________________________________')
print()
print()


for directory in dirs:
    print('directory:',directory)
    #print('files in directory:',glob.glob(directory + '*.txt'))
    print()
    
    # check if directory has not been processed before
    if '.pdf' in ''.join(glob.glob(directory + '*')):
        print('nothing to do')
        pass
    # otherwise create files
    else:
        tuple_params = get_params()
        
        process_statistics(directory,*tuple_params)
        
        process_frames(directory,*tuple_params)
    # end of loop step
    
    print('__________________________________________________________________________________________________________________')
    print()
    print()
    #



# przemyslec to!!!
"""
files_bin_wf = glob.glob('./bin/*.bin')
for f in files_bin_wf:
    data = np.memmap(f, dtype='complex128', mode='r')
    x = np.linspace(-0.5,0.5,len(data))
    plt.plot(x,np.real(data),label='Re')
    plt.plot(x,np.imag(data),label='Im')
    plt.plot(x,np.abs(data),label=r'$|/psi|$')
    plt.grid(True)
    plt.title(f)
    plt.legend(loc='lower left')
    figname = f.replace('.bin','.pdf')
    figname = figname.replace('./bin/','./')
    print(f,'- figure saved in:',figname)
    plt.savefig(figname)
    plt.clf()
    """