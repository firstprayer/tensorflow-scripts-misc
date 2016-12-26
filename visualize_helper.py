from matplotlib import pyplot as plt
import numpy as np

def plot_images(images, row, col, save_filename=None):
  fig, axarr = plt.subplots(row, col)
  for r in range(row):
    for c in range(col):
      img = images[r * col + c, :, :]
      axarr[r, c].imshow(img, cmap='gray')

  if save_filename:
    plt.savefig(save_filename)
  else:
    plt.show()
  plt.clf()
