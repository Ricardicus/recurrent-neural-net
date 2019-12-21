import numpy as np
from matplotlib import pyplot as pl
import sys

if __name__=="__main__":
  
  if len(sys.argv) == 2:
    name = sys.argv[1]

    with open(name) as f:
      lines = f.readlines()

      iterations = []
      losses = []

      for line in lines:
        try:
          line.split(',')[1]
          iterations.append(int(line.split(',')[0]))
          losses.append(float(line.split(',')[1]))
        except IndexError:
          pass

      pl.plot(iterations, losses)
      pl.show()
  else :
    print("Usage: python ", sys.argv[0], " csv-file")