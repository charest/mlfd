import argparse
import matplotlib.pyplot as plt
from configparser import ConfigParser

from shock import Shock1D
from sod import Sod1D

###############################################################################

def main():
  
  parser = argparse.ArgumentParser(description="Simulater.")
  parser.add_argument("--config", required=True, type=str, help="Config file to use.")
  parser.add_argument("--freq", default=10, type=int, help="Plot frequency.")
  parser.add_argument("--solver", type=str, help="The solver to use.")
      
  args = parser.parse_args()

  if args.solver == "shock":
    s = Shock1D(args.config)
  elif args.solver == "sod":
    s = Sod1D(args.config)

  
  fig, ax = plt.subplots(2, 2, figsize=(10, 8))

  it = 0
  t = 0
      
  lbl = "t={:.2e}".format(t)
  ax[0,0].plot(s.xc, s.dc, marker='x', label=lbl)
  ax[0,1].plot(s.xc, s.vc, marker='x', label=lbl)
  ax[1,0].plot(s.xc, s.pc, marker='x', label=lbl)
  ax[1,1].plot(s.xc, s.ec, marker='x', label=lbl)
    
  cfg = ConfigParser()
  cfg.read(args.config)
  tmax = cfg.getfloat('case', 'tmax')
  num_iter = cfg.getint('case', 'num_iter')
  
  while (it < num_iter and t < tmax):

      dtmax = tmax - t
      dt = s.advance(dtmax)

      t += dt
      it += 1
     
      print(f"it={it}, dt={dt}, t={t}")
  
      if it % args.freq == 0:
        lbl = "t={:.2e}".format(t)
        ax[0,0].plot(s.xc, s.dc, marker='x', label=lbl)
        ax[0,1].plot(s.xc, s.vc, marker='x', label=lbl)
        ax[1,0].plot(s.xc, s.pc, marker='x', label=lbl)
        ax[1,1].plot(s.xc, s.ec, marker='x', label=lbl)
  
  
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


if __name__ == "__main__":
  main()
      
