"""
Program calculates the exact solution to Sod-shock tube class
problems -- namely shock tubes which produce shocks, contact
discontinuities, and rarefraction waves.

Solution is computed at locations x at time t. (Though   
due to self-similarity, the exact solution is identical for
identical values of x/t). Output to the file 'exact_sod.out'
is in the format

     x position, density, velocity, pressure

NOTE : Since the post-shock flow is nonadiabatic, whereas
the flow inside the rarefaction fan is adiabatic, the problem
is not left-right symmetric. In particular, the high-density
initial state MUST BE input on the left side.

Originally written by Robert Fisher, 12/5/96.
"""

import numpy as np
from scipy import optimize
from configparser import ConfigParser

def timestep(dx, vc, ac):
    nc = len(dx)
    dti = 0.
    for i in range(nc):
        ss = abs(vc[i]) + ac[i]
        dti = max(dti, ss/dx[i])
    return 1. / dti

def sod_exact(gamma, pl, rhol, pr, rhor, xs, xmid, t):

  mu2  = (gamma - 1.) / (gamma + 1.)
  
  # Define sound speeds for the left and right sides of tube.
  
  cl = np.sqrt (gamma * pl / rhol)
  cr = np.sqrt (gamma * pr / rhor)
  
  # Solve for the postshock pressure pm.
  
  def func(pm):
      return -2*cl*(1 - (pm/pl)**((-1 + gamma)/(2*gamma))) / -(cr*(-1 + gamma)) - (-1 + pm/pr)*((1 - mu2)/(gamma*(mu2 + pm/pr)))**0.5
  
  pm = optimize.bisect(func, pr, pl, xtol=1.e-16)
  #pm = rtbis (pr, pl, 1.e-16)
  
  # Define the density to the left of the contact discontinuity rhoml.
   
  rhoml = rhol * (pm / pl) ** (1. / gamma)
  
  # Define the postshock fluid velocity vm.
  
  vm = 2. * cl / (gamma - 1.) * (1. - (pm / pl) ** ( (gamma - 1.) / (2. * gamma) ))
  
  # Define the postshock density rhomr.
  
  rhomr = rhor *  ( (pm + mu2 * pr) / (pr + mu2 * pm) )
  
  # Define the shock velocity vs.
  
  vs = vm / (1. - rhor / rhomr) 
  
  # Define the velocity of the rarefraction tail, vt.
  
  vt = cl - vm / (1. - mu2) 
  
  # Output tables of density, velocity, and pressure at time t.
  numcells = len(xs)
  density = np.empty(numcells)
  pressure = np.empty(numcells)
  velocity = np.empty(numcells)
  
  for i in range(numcells):
     
      x = xs[i] - xmid
   
      if (x <= - cl * t):
          density[i] = rhol
      elif (x <= -vt * t):
          density[i] = rhol * (-mu2 * (x / (cl * t) ) + (1 - mu2) ) ** (2. / (gamma - 1.)) 
      elif (x <= vm * t):
          density[i] = rhoml
      elif (x <= vs * t):
          density[i] = rhomr
      else:
          density[i] = rhor
      
      if (x <= - cl * t):
          pressure[i] = pl
      elif (x <= -vt * t):
          pressure[i] = pl * (-mu2 * (x / (cl * t) ) + (1 - mu2) ) ** (2. * gamma / (gamma - 1.))
      elif (x <= vs * t):
          pressure[i] = pm 
      else:            
          pressure[i] = pr
      
      if (x <= -cl * t):
          velocity[i] = 0.0
      elif (x <= -vt * t):
          velocity[i] = (1 - mu2) * (x / t + cl)
      elif (x <= vs * t):
          velocity[i] = vm
      else: 
          velocity[i] = 0.0

  return density, velocity, pressure


class Sod1D:

  def __init__(self, config):

    cfg = ConfigParser()
    cfg.read(config)
    
    self.gamma= 1.4
    xmin = cfg.getfloat('case', 'xmin')
    xmax = cfg.getfloat('case', 'xmax')
    self.xmid = cfg.getfloat('case', 'xmid')
    self.dl = cfg.getfloat('case', 'dl')
    self.pl = cfg.getfloat('case', 'pl')
    self.dr = cfg.getfloat('case', 'dr')
    self.pr = cfg.getfloat('case', 'pr')
    
    self.cfl = cfg.getfloat('case', 'cfl')
    num_cells = cfg.getint('case', 'num_cells')
   
    self.t = 0
    
    self.xn = np.linspace(xmin, xmax, num_cells+1)
    self.dx = np.diff(self.xn)
    self.xc = self.xn[:num_cells] + self.dx/2
       
    self.dc, self.vc, self.pc = sod_exact(self.gamma, self.pl, self.dl, self.pr, self.dr, self.xc, self.xmid, self.t)
    self.ec = self.pc / self.dc / (self.gamma - 1.) 
  
  def advance(self, dtmax):
  
      ac = np.sqrt( self.gamma * self.pc / self.dc )

      dt = self.cfl * timestep(self.dx, self.vc, ac)
      dt = min(dt, dtmax)

      self.t += dt
      
      self.dc, self.vc, self.pc = sod_exact(self.gamma, self.pl, self.dl, self.pr, self.dr, self.xc, self.xmid, self.t)
      self.ec = self.pc / self.dc / (self.gamma - 1.)

      return dt

  def regrid(self, fact):

    nf = len(self.xc)
    nc = nf // fact

    geom = np.full(fact, 1./fact)
    xc = np.empty(nc)
    dc = np.empty(nc)
    vc = np.empty(nc)
    ec = np.empty(nc)

    mom = self.dc * self.vc
    en = self.dc * (self.ec + 0.5*self.vc*self.vc)
    
    for i in range(nc):
      start = i * fact
      end = start + fact
      xc[i] = np.dot( self.xc[start:end], geom )
      dc[i] = np.dot( self.dc[start:end], geom )
      vc[i] = np.dot( mom[start:end], geom ) / dc[i]
      ec[i] = np.dot( en [start:end], geom ) / dc[i] - 0.5*vc[i]*vc[i]
    
    pc = (self.gamma-1) * dc * ec
    
    return xc, dc, vc, ec, pc 


"""

  10  format (E22.16, ' ', E22.16, ' ', E22.16, ' ', E22.16)  
                    
      End



      function func (pm)

*//////////////////////////////////////////////////////////////////////
*/
*/  func is obtained from an identity matching the post-shocked      
*/  pressure to the post-rarefraction pressure (true since there is
*/  no pressure jump across the contact discontinuity). We use it to
*/  numerically solve for pm given the left and right initial states.
*/
*//////////////////////////////////////////////////////////////////////
 

      implicit none

      real*8 func, pm 
      real*8 gamma, mu2
 
      parameter (gamma   =    1.4)
      parameter (mu2     =   (gamma - 1.) / (gamma + 1.))
   
      real*8 pl, pr, rhol, rhor, cl, cr
 
      common/block1/ pl, pr, rhol, rhor, cl, cr


      func = -2*cl*(1 - (pm/pl)**((-1 + gamma)/(2*gamma)))/
     &    -   (cr*(-1 + gamma)) + 
     &    -  (-1 + pm/pr)*((1 - mu2)/(gamma*(mu2 + pm/pr)))**0.5
      return

      end 
  
      FUNCTION rtbis(x1,x2,xacc)

*/////////////////////////////////////////////////////////////////////////
*/
*/ rtbis is borrowed from Numerical Recipes. It is a bisection algorithm,
*/ which we use to solve for pm using a call to func.
*/
*/ Note that the arguments to rtbis have been altered and the value of
*/ JMAX increased. Otherwise, it is identical to the NR version.
*/
*/////////////////////////////////////////////////////////////////////////

      INTEGER JMAX
      REAL*8 rtbis,x1,x2,xacc,func
      EXTERNAL func
      PARAMETER (JMAX=100)
      INTEGER j
      REAL*8 dx,f,fmid,xmid
      fmid=func(x2)
      f=func(x1)
      if(f*fmid.ge.0.) then
        print *, 'root must be bracketed in rtbis'
        stop
      endif
      if(f.lt.0.)then
        rtbis=x1
        dx=x2-x1
      else
        rtbis=x2
        dx=x1-x2
      endif
      do 11 j=1,JMAX
        dx=dx*5.D-1
        xmid=rtbis+dx
        fmid=func(xmid)
        if(fmid.le.0.)rtbis=xmid
        if(dabs(dx).lt.xacc .or. fmid.eq.0.) return
11    continue
      print *, 'too many bisections in rtbis'
      END
"""
