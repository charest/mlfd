import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser
import argparse
        
def flux(u):
    return 0.5*u*u

def rusanov(ul, ur):
    a = max(abs(ul), abs(ur))
    fl = flux(ul)
    fr = flux(ur)
    return 0.5 * (fl + fr) - 0.5 * a * (ur - ul)

def timestep(dx, vc):
    nc = len(dx)
    dti = 0.
    for i in range(nc):
        dti = max(dti, abs(vc[i])/dx[i])
    return 1. / dti

def fv(dx, vc):
    nc = len(dx)
    
    du = np.zeros(nc)

    for i in range(1,nc-1):
        fl = rusanov(vc[i-1], vc[i])
        fr = rusanov(vc[i], vc[i+1])
        du[i] = -(fr - fl) / dx[i]

    return du
###############################################################################
    
parser = argparse.ArgumentParser(description="Benchmark matrix multiplication algorithms.")
parser.add_argument("--config", required=True, type=str, help="Config file to use.")
parser.add_argument("--freq", default=10, type=int, help="Plot frequency.")
    
args = parser.parse_args()
    
config = ConfigParser()
config.read(args.config)

num_cells = config.getint('case', 'num_cells')
xmin = config.getfloat('case', 'xmin')
xmax = config.getfloat('case', 'xmax')
xmid = config.getfloat('case', 'xmid')
vl = config.getfloat('case', 'vl')
vr = config.getfloat('case', 'vr')
cfl = config.getfloat('case', 'cfl')
tmax = config.getfloat('case', 'tmax')

num_iter = config.getint('case', 'num_iter')

xn = np.linspace(xmin, xmax, num_cells+1)
dx = np.diff(xn)
xc = xn[:num_cells] + dx/2

vc = np.where( xc < xmid, vl, vr)

t = 0.
it = 0

lbl = "t={:.2e}".format(t)
plt.plot(xc, vc, marker='o')


while (it < num_iter and t < tmax):
   
    t_left = tmax - t

    dt = timestep(dx, vc)
    dt = cfl * min(dt, t_left)
    du = fv(dx, vc)
    vc += du * dt
    t += dt
    it += 1
    print(f"it={it}, dt={cfl*dt}, t={t}")
    
    if it % args.freq == 0:
      lbl = "t={:.2e}".format(t)
      plt.plot(xc, vc, marker='x', label=lbl)


# Plot the array
plt.xlabel("x")
plt.ylabel("v")
plt.grid(True)
plt.show()


