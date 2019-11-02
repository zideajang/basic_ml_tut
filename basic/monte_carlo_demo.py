#coding=utf-8

import numpy as np

num_iterations = 10**4

pts = np.random.uniform(low=-1,high=1,size=(num_iterations,2))

sq_radiuses = np.sum(pts**2,axis=1)
in_circle = (sq_radiuses < 1)
approx_pi = 4 * np.sum(in_circle) /(num_iterations) * 1.00000

print(approx_pi)