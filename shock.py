import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser
import argparse

GAMMA = 1.4
        
def flux(d, u, e, p):
    E = e + 0.5*u*u
    return d*u, d*u*u + p, u*(d*E + p)

def conserved(d, u, e):
    E = e + 0.5*u*u
    return d, d*u, d*E

def soundspeed(dc, pc):
    return np.sqrt( GAMMA * pc / dc )

def energy(dc, pc):
    return pc / dc / (GAMMA-1)

def pressure(dc, ec):
    return (GAMMA-1) * dc * ec

def rusanov(dl, ul, el, pl, al, dr, ur, er, pr, ar):
    smax = max(abs(ul), abs(ur)) + max(al, ar)
    Fl = np.array( flux(dl, ul, el, pl) )
    Fr = np.array( flux(dr, ur, er, pr) )
    Ul = np.array( conserved(dl, ul, el) )
    Ur = np.array( conserved(dr, ur, er) )
    return 0.5 * (Fl + Fr) - 0.5 * smax * (Ur - Ul)

def timestep(dx, vc, ac):
    nc = len(dx)
    dti = 0.
    for i in range(nc):
        ss = abs(vc[i]) + ac[i]
        dti = max(dti, ss/dx[i])
    return 1. / dti

def fv(dx, dc, vc, ec, pc, ac):
    nc = len(dx)
    
    dd = np.zeros(nc)
    dv = np.zeros(nc)
    de = np.zeros(nc)

    for i in range(1,nc-1):
        fl = rusanov(dc[i-1], vc[i-1], ec[i-1], pc[i-1], ac[i-1], dc[i  ], vc[i  ], ec[i  ], pc[i  ], ac[i  ])
        fr = rusanov(dc[i  ], vc[i  ], ec[i  ], pc[i  ], ac[i  ], dc[i+1], vc[i+1], ec[i+1], pc[i+1], ac[i+1])
        dd[i], dv[i], de[i] = -(fr - fl) / dx[i]

    return dd, dv, de

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
dl = config.getfloat('case', 'dl')
vl = config.getfloat('case', 'vl')
pl = config.getfloat('case', 'pl')
dr = config.getfloat('case', 'dr')
vr = config.getfloat('case', 'vr')
pr = config.getfloat('case', 'pr')
cfl = config.getfloat('case', 'cfl')
tmax = config.getfloat('case', 'tmax')

num_iter = config.getint('case', 'num_iter')

xn = np.linspace(xmin, xmax, num_cells+1)
dx = np.diff(xn)
xc = xn[:num_cells] + dx/2

dc = np.where( xc < xmid, dl, dr)
vc = np.where( xc < xmid, vl, vr)
pc = np.where( xc < xmid, pl, pr)
ec = energy(dc, pc)

fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    
t = 0.
it = 0

lbl = "t={:.2e}".format(t)
ax[0,0].plot(xc, dc, marker='x', label=lbl)
ax[0,1].plot(xc, vc, marker='x', label=lbl)
ax[1,0].plot(xc, pc, marker='x', label=lbl)
ax[1,1].plot(xc, ec, marker='x', label=lbl)

while (it < num_iter and t < tmax):
   
    t_left = tmax - t

    ac = soundspeed(dc, pc)
    dt = cfl * timestep(dx, vc, ac)
    dt = min(dt, t_left)
    dd, dv, de = fv(dx, dc, vc, ec, pc, ac)
    dc += dd * dt
    vc += dv * dt
    ec += de * dt
    pc = pressure(dc, ec)
    t += dt
    it += 1
    print(f"it={it}, dt={cfl*dt}, t={t}")

    if it % args.freq == 0:
      lbl = "t={:.2e}".format(t)
      ax[0,0].plot(xc, dc, marker='x', label=lbl)
      ax[0,1].plot(xc, vc, marker='x', label=lbl)
      ax[1,0].plot(xc, pc, marker='x', label=lbl)
      ax[1,1].plot(xc, ec, marker='x', label=lbl)


# Plot the array
ax[0,0].legend()
ax[0,0].set_title("Density")
ax[0,1].set_title("Velocity")
ax[1,0].set_title("Pressure")
ax[1,1].set_title("Energy")
ax[1,0].set_xlabel("x")
ax[1,1].set_xlabel("x")
for a in ax.flat:
  a.grid(True)
plt.show()


