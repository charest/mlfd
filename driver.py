import argparse
from shock import Shock1D
from sod import Sod1D
from configparser import ConfigParser
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

GAMMA = 1.4

def soundspeed(dc, pc):
    return np.sqrt( GAMMA * pc / dc )
def energy(dc, pc):
    return pc / dc / (GAMMA-1)
def pressure(dc, ec):
    return (GAMMA-1) * dc * ec


def timestep(dx, vc, ac):
    nc = len(dx)
    dti = 0.
    for i in range(nc):
        ss = abs(vc[i]) + ac[i]
        dti = max(dti, ss/dx[i])
    return 1. / dti

def add_training_data(dt, xc, vals_in, vals_out, xdata, ydata):
  nv = len(vals_in)
  n = vals_in[0].shape[0]
  for i in range(1,n-1):
    di = [dt, xc[i]-xc[i-1]]
    do = []
    for v in range(nv):
      di.append( vals_in[v][i-1]  )
      di.append( vals_in[v][i]    )
      di.append( vals_in[v][i+1]  )
      do.append( vals_out[v][i] )
    ydata.append(do)
    xdata.append(di)

###############################################################################
class Model1D:

  def __init__(self, config, model):

    self.model = model

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

      ncell = len(self.dc)
  
      ac = soundspeed(self.dc, self.pc)
      dt = self.cfl * timestep(self.dx, self.vc, ac)
      dt = min(dt, dtmax)

      vpack = (self.dc, self.vc, self.ec, self.pc)

      x = []
      for i in range(1, ncell-1):
        dx = self.xc[i] - self.xc[i-1]
        xi = [dt, dx]
        for v in vpack:
          xi.extend([v[i-1], v[i], v[i+1]])
        x.append(xi)
      y = self.model.predict(x)
      for i in range(1,ncell-1):
        self.dc[i] = y[i-1][0]
        self.vc[i] = y[i-1][1]
        self.ec[i] = y[i-1][2]
        self.pc[i] = y[i-1][3]
      
      #self.pc = pressure(self.dc, self.ec)

      return dt



parser = argparse.ArgumentParser(description="Main driver.")
parser.add_argument("--config", required=True, type=str, help="Config file to use.")
parser.add_argument("--fact", default=2, type=int, help="Regrid factor.")
parser.add_argument("--solver", required=True, type=str, help="The solver to use.")
parser.add_argument("--freq", default=10, type=int, help="Plot frequency.")
parser.add_argument("--config2", required=True, type=str, help="Config file to use.")
    
args = parser.parse_args()
  
if args.solver == "shock":
  s = Shock1D(args.config)
elif args.solver == "sod":
  s = Sod1D(args.config)
else:
  raise RuntimeError(f"Unknown solver: {args.solver}")
  
it = 0
t = 0

cfg = ConfigParser()
cfg.read(args.config)
tmax = cfg.getfloat('case', 'tmax')
num_iter = cfg.getint('case', 'num_iter')
    
xc, dc, vc, ec, pc = s.regrid(args.fact)
  
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
lbl = "t={:.2e}".format(t)
ax[0,0].plot(xc, dc, marker='x', label=lbl)
ax[0,1].plot(xc, vc, marker='x', label=lbl)
ax[1,0].plot(xc, pc, marker='x', label=lbl)
ax[1,1].plot(xc, ec, marker='x', label=lbl)

xdata = []
ydata = []

while (it < num_iter and t < tmax):
    
    t0 = t
    xc0 = xc
    dc0 = dc
    vc0 = vc
    ec0 = ec
    pc0 = pc

    for n in range(args.fact):
      if (it < num_iter and t < tmax):
        dtmax = tmax - t
        dt = s.advance(dtmax)
        t += dt
        it += 1
      
    dtc = t - t0    
    xc, dc, vc, ec, pc = s.regrid(args.fact)
    add_training_data(dtc, xc, (dc0, vc0, ec0, pc0), (dc, vc, ec, pc), xdata, ydata)

    if it % args.freq == 0:
      lbl = "t={:.2e}".format(t)
      ax[0,0].plot(xc, dc, marker='x', label=lbl)
      ax[0,1].plot(xc, vc, marker='x', label=lbl)
      ax[1,0].plot(xc, pc, marker='x', label=lbl)
      ax[1,1].plot(xc, ec, marker='x', label=lbl)


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.2, random_state=42)

#model = LinearRegression()
#model = MLPRegressor(hidden_layer_sizes=(100, 100, 100), activation='relu', solver='adam', random_state=42, alpha=0.001, tol=1e-6)

# Define the degree of the polynomial
degree = 1

model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=degree)),
    ('linear_regression', LinearRegression())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print("R^2 Score:", model.score(X_test, y_test))

#print(f"Coefficients: {model.coef_}")
#print(f"Intercept: {model.intercept_}")


sm = Model1D(args.config2, model)

cfg = ConfigParser()
cfg.read(args.config2)
tmax = cfg.getfloat('case', 'tmax')
num_iter = cfg.getint('case', 'num_iter')


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
#plt.show()
  



it = 0
t = 0

fig, ax = plt.subplots(2, 2, figsize=(10, 8))
lbl = "t={:.2e}".format(t)
ax[0,0].plot(sm.xc, sm.dc, marker='x', label=lbl)
ax[0,1].plot(sm.xc, sm.vc, marker='x', label=lbl)
ax[1,0].plot(sm.xc, sm.pc, marker='x', label=lbl)
ax[1,1].plot(sm.xc, sm.ec, marker='x', label=lbl)

num_iter = 1
while (it < num_iter and t < tmax):

    dtmax = tmax - t
    dt = sm.advance(dtmax)
    t += dt
    it += 1
    print(it, t)
      
    #if it % args.freq == 0:
    lbl = "t={:.2e}".format(t)
    ax[0,0].plot(sm.xc, sm.dc, marker='x', label=lbl)
    ax[0,1].plot(sm.xc, sm.vc, marker='x', label=lbl)
    ax[1,0].plot(sm.xc, sm.pc, marker='x', label=lbl)
    ax[1,1].plot(sm.xc, sm.ec, marker='x', label=lbl)


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
