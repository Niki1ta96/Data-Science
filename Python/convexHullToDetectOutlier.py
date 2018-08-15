# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 08:41:07 2018

@author: nikita.jaikar
"""

from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt

points = np.random.rand(30,2)
hull = ConvexHull(points)
plt.plot(points[:,0], points[:,1],0)

for simplex in hull.simplices:
    plt.plot(points[simplex,0],points[simplex,1],'k-')

