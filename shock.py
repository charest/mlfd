import numpy as np
from configparser import ConfigParser

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
class Shock1D:

  def __init__(self, config):

    cfg = ConfigParser()
    cfg.read(config)
    
    num_cells = cfg.getint('case', 'num_cells')
    xmin = cfg.getfloat('case', 'xmin')
    xmax = cfg.getfloat('case', 'xmax')
    xmid = cfg.getfloat('case', 'xmid')
    dl = cfg.getfloat('case', 'dl')
    vl = cfg.getfloat('case', 'vl')
    pl = cfg.getfloat('case', 'pl')
    dr = cfg.getfloat('case', 'dr')
    vr = cfg.getfloat('case', 'vr')
    pr = cfg.getfloat('case', 'pr')

    self.cfl = cfg.getfloat('case', 'cfl')
  
    self.xn = np.linspace(xmin, xmax, num_cells+1)
    self.dx = np.diff(self.xn)
    self.xc = self.xn[:num_cells] + self.dx/2
  
    self.dc = np.where( self.xc < xmid, dl, dr)
    self.vc = np.where( self.xc < xmid, vl, vr)
    self.pc = np.where( self.xc < xmid, pl, pr)
    self.ec = energy(self.dc, self.pc)

  def advance(self, dtmax):
  
      ac = soundspeed(self.dc, self.pc)
      dt = self.cfl * timestep(self.dx, self.vc, ac)
      dt = min(dt, dtmax)
      dd, dv, de = fv(self.dx, self.dc, self.vc, self.ec, self.pc, ac)
      self.dc += dd * dt
      self.vc += dv * dt
      self.ec += de * dt
      self.pc = pressure(self.dc, self.ec)

      return dt



