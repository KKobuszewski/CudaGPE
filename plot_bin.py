import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob
import math
import string


# simulation parameters
xmax = .5
xmin = -0.5
dt = 1e-07




files_bin_wf = glob.glob('./bin/*.bin')
for f in files_bin_wf:
    data = np.memmap(f, dtype='complex128', mode='r')
    x = np.linspace(xmin,xmax,len(data))
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


files = glob.glob('./data*/stat*.txt')
print('statistics:',files)

for f in files:
    print(f)
    data = np.loadtxt(f, skiprows=5)
    t = data[:,0]
    n = data[:,1]
    mu = data[:,2]
    T = data[:,3]
    Vext = data[:,4]
    Vcon = np.zeros(len(Vext))
    Vdip = np.zeros(len(Vext))
    
    virial = 2*T - 2*Vext
    Etot = T + Vext
    
    print('norm change',np.max(n) - np.min(n),'per',np.max(t),'steps (',(np.max(n) - np.min(n))/np.max(t),')')
    print('kinetic energy  mean:',np.mean(T),'  min:',np.min(T),'  std. dev:',np.std(T))
    print('ext. pot. energy  mean:',np.mean(Vext),'  min:',np.min(Vext),'  std. dev:',np.std(Vext))
    print('total energy  mean:',np.mean(Etot),'  min:',np.min(Etot),'  std. dev:',np.std(Etot))
    print('virial  mean:',np.mean(virial),'  min:',np.min(virial),'  std. dev:',np.std(virial))
    
    stats_str = ('norm change: {} per {} steps ({} per step)\n\n'.format(np.max(n) - np.min(n),np.max(t),(np.max(n) - np.min(n))/np.max(t)),
             'kinetic energy\nmean: {}   min:{}   std. dev.:{}\n\n'.format(np.mean(T),np.min(T),np.std(T)),
             'ext. pot. energy\nmean: {}   min:{}   std. dev.:{}\n\n'.format(np.mean(Vext),np.min(Vext),np.std(Vext)),
             'total energy\nmean: {}   min:{}   std. dev.:{}\n\n'.format(np.mean(Etot),np.min(Etot),np.std(Etot)),
             'virial\nmean: {}   min:{}   std. dev.:{}\n\n'.format(np.mean(virial),np.min(virial),np.std(virial))
             )
    stats_str = ''.join(stats_str)
             
             
    
    figname = f.replace('.txt','.pdf')
    with PdfPages(figname) as pdf:
        
        fig = plt.figure(figsize=(12,4),dpi=500)
        plt.title('statistics:')
        plt.axis('off')
        plt.text(0,0,stats_str)
        pdf.savefig(fig)
        plt.close()
        
        plt.margins(1)
        fig = plt.figure(figsize=(8,8),dpi=1000)
        plt.grid(True)
        plt.title('mean values of different energy operators')
        plt.xlabel('timesteps')
        plt.ylabel('energy')
        plt.plot(t,T,label='T')
        plt.plot(t,Vext,label=r'$V_{ext}$')
        plt.plot(t,(T+Vext)/2,label='srednia')
        plt.legend()
        
        plt.yticks(rotation='vertical')
        #plt.yticks(  np.arange(np.min(T)*1.1,np.max(T)*1.1,4), [ item for item in (np.arange(np.min(T)*1.1,np.max(T)*1.1,4)).astype('|S6') ], rotation='vertical'  )
        ax = plt.gca()
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        #ax.set_yticks(np.linspace(np.min(T)*1.1,np.max(T)*1.1,4))
        #ax.set_yticklabels([ item for item in (np.linspace(np.min(T)*1.1,np.max(T)*1.1,4)).astype('|S6') ])
        #plt.tick_params(axis='y',labelsize=6)
        #fig.canvas.draw()
        plt.tight_layout()
        plt.subplots_adjust()
        pdf.savefig(fig)
        plt.close()
        
        plt.rc('text',usetex=True)
        fig = plt.figure(figsize=(8,8),dpi=1000)
        plt.grid(True)
        plt.title('virial and changes of total energy')
        plt.plot(t,virial,label='Virial')
        plt.plot(t,Etot - np.min(Etot),label=r'$E_{tot} - E_{min}$')
        plt.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        plt.rc('text',usetex=True)
        
        # here add some more PdfPages
        

#
