from numpy import *
import numpy as np
import saffine.detrending_method as dm
# from numba import jit

# @jit

def multi_detrending(y,step_size,q,order) :
	# y: input data,stored as a row or column vector
	# q: q spectrum

	#q = [-1 -0.1 0.1 1,2,3,4,5];
	#step_size = 1
	q = mat(q)
	len = shape(y)[1]
	imax = int(round(log2(len)))
	#order = 2
	result = mat(np.zeros((shape(q)[1] + 1 , int((imax - 2)/step_size) + 1)))
	k = 1
	for i in range(1 , imax , step_size) :
		w = int(round(2 ** i + 1))
		if w / 2 == 1 :
			w  = w + 1
		detrended_data , trend = dm.detrending_method(y , w , order)
		result[0 , k-1] = (w + 1)/2
		for j in range(1,shape(q)[1] + 1):
			#Euclidean norm
			abs_detrended_data = power(abs(detrended_data) , q[0 , j-1])
			Sum = abs_detrended_data.sum(axis = 0) / (shape(detrended_data)[0] - 1)

			result[j , k-1] = Sum[0 , 0] ** (1 / q[0 , j - 1])

			# result(j + 1,k) = (sum(abs(detrend_data - mean(detrended_data))) **q[j-1] / ((shape(detrended_data)[0] - 1) ** (1/q[j-1]))
			# earlier analysis suggests that without removing mean yield
			# more accurate estimate of H values

		k = k + 1

	result = log2(result)

	return result
