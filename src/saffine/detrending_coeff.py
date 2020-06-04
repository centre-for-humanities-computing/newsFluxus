from numpy import *
import numpy as np
# from numba import jit

# @jit

def detrending_coeff(win_len , order):

#win_len = 51
#order = 2
	n = (win_len-1)/2
	A = mat(ones((win_len,order+1)))
	x = np.arange(-n , n+1)
	for j in range(0 , order + 1):
		A[:,j] = mat(x ** j).T

	coeff_output = (A.T * A).I * A.T
	return coeff_output , A

# coeff_output,A = detrending_coeff(5,2)
# print(coeff_output)
# print(A)
