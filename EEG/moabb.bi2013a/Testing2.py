# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 16:18:10 2022

@author: antona
"""

# import module
from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt
  
# assign images
img1 = Image.open("C:\\Temp\\Figure1.png")
img2 = Image.open("C:\\Temp\\Figure2.png")
  
# finding difference
diff = ImageChops.difference(img1, img2)
  
# showing the difference
diff.show()

histogram, bin_edges = np.histogram(imave1, bins=256, range=(0, 1))
histogram, bin_edges = np.histogram(imave2, bins=256, range=(0, 1))

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixel count")
plt.xlim([0.0, 1.0])  # <- named arguments do not work here

plt.plot(bin_edges[0:-1], histogram)  # <- or here
plt.show()