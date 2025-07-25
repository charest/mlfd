import numpy as np
import matplotlib.pyplot as plt
        
def flux(u):
    return u*u

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
    #a = vc[:-1]**2
    #b = vc[1:]**2
    #cond = vc[:-1] < vc[1:]
    #
    #x = np.where(cond, np.minimum(a, b), np.maximum(a, b))

    #vr = vc[1:] - vc[:-1]

    du = np.zeros(nc)

    for i in range(1,nc-1):
        fl = rusanov(vc[i], vc[i-1])
        fr = rusanov(vc[i], vc[i+1])
        du[i] = (fr - fl) / dx[i]

    return du

num_cells = 100
xlim = (0, 1)
xmid = 0.5
vl = 1.
vr = 2.
cfl = 1.

num_iter = 1

xn = np.linspace(*xlim, num_cells+1)
dx = np.diff(xn)
xc = xn[:num_cells] + dx/2


vc = np.where( xc < xmid, vl, vr)


for it in range(num_iter):

    dt = timestep(dx, vc)
    print(dt)
    du = fv(dx, vc)
    vc += du * dt


# Plot the array
plt.plot(xc, vc, marker='o')
plt.xlabel("x")
plt.ylabel("v")
plt.grid(True)
plt.show()


