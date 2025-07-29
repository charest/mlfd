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
import matplotlib.pyplot as plt

# xmax determines the size of the computational domain (-xmax, +xmax).
# numcells determines the number of cells in the output table.          

gamma   = 1.4
mu2     = (gamma - 1.) / (gamma + 1.)
xmin  = 0.
xmax 	= 2.
xmid = (xmax - xmin) / 2.
numcells= 500
     
# Define the time of the problem.

t = 2.45e-1
 
# Define the Sod problem initial conditions for the left and right states.

pl = 1.
pr = 1.e-1

rhol = 1.
rhor = 1.25e-1      

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
density = np.empty(numcells)
pressure = np.empty(numcells)
velocity = np.empty(numcells)
coords = np.empty(numcells)
xdelta = xmax - xmin

for i in range(numcells):
   
    x = xmin + xdelta * i / numcells
    coords[i] = x

    x -= xmid
 
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

energy = pressure / density / (gamma - 1.)
    
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax[0, 0].plot(coords, density)
ax[0, 0].set_title('Density')

ax[0, 1].plot(coords, velocity)
ax[0, 1].set_title('Velocity')

ax[1, 0].plot(coords, pressure)
ax[1, 0].set_title('Pressure')

ax[1, 1].plot(coords, energy)
ax[1, 1].set_title('Energy')

#ax[1, 1].plot(x, -y)
#ax[1, 1].set_title('Plot 4')

# Improve layout
plt.tight_layout()
plt.show()



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
