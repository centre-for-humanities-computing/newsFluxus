from numpy import *
import numpy as np
import saffine.detrending_coeff as dc
# from numba import jit

# @jit
def detrending_method(data , seg_len , fit_order) :
	nrows,ncols = shape(data)
	if nrows < ncols :
		data = data.T

	# seg_len = 1001,odd number
	nonoverlap_len = int((seg_len - 1) / 2)
	data_len = shape(data)[0]
	# calculate the coefficient,given a window size and fitting order
	coeff_output , A = dc.detrending_coeff(seg_len , fit_order)
	A_coeff = A * coeff_output

	for seg_index in range(1 , 2) :
		#left trend
		#seg_index = 1

		xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) , seg_index * (seg_len - 1) + 2)
		xi_left = mat(xi)
		xi_max = xi.max()
		xi_min = xi.min()
		seg_data = data[xi_min - 1 : xi_max , 0]
		left_trend = (A_coeff * seg_data).T

		# mid trend

		if seg_index * (seg_len - 1) + 1 + nonoverlap_len > data_len :
			xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) + nonoverlap_len , data_len + 1)
			xi_mid = mat(xi)
			xi_max = xi.max()
			xi_min = xi.min()
			seg_data = data[xi_min - 1 : xi_max , 0]
			nrows_seg = shape(seg_data)[0]

			if nrows_seg < seg_len :
				coeff_output1 , A1 = dc.detrending_coeff(nrows_seg , fit_order)
				A_coeff1 = A1 * coeff_output1
				mid_trend = (A_coeff1 * seg_data).T
			else :
				mid_trend = (A_coeff * seg_data).T

			xx1 = left_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
			xx2 = mid_trend[0 , 0 : int((seg_len + 1) / 2)]
			w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
			xx_left = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

			record_x = xi_left[0 , 0 : nonoverlap_len]
			record_y = left_trend[0 , 0 : nonoverlap_len]
			mid_start_index = mat([(j) for j in range(shape(xi_mid)[1]) if xi_mid[0 , j] == xi_left[0 , shape(xi_left)[1] - 1] + 1])
			nrows_mid = shape(mid_start_index)[0]
			mid_start_index = mid_start_index[0 , 0]

			if nrows_mid == 0 :
				record_x = hstack((record_x , xi_left[0 , int((shape(xi_left)[1] + 3) / 2)-1 : shape(xi_left)[1]]))
				record_y = hstack((record_y , xx_left[0 , 1 : shape(xx_left)[1]]))
			else :
				record_x = hstack((record_x , xi_left[0 , int((shape(xi_left)[1] + 1)/ 2 )-1 : shape(xi_left)[1]] , xi_mid[0 , mid_start_index : shape(xi_mid)[1]]))
				record_y = hstack((record_y , xx_left[0 : shape(xx_left)[1]] , mid_trend[0 , int((seg_len + 3) / 2) - 1 : shape(mid_trend)[1]]))

			detrended_data = data - record_y.T

			return  detrended_data, record_y

		else :
			xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) + nonoverlap_len , seg_index * (seg_len - 1) + nonoverlap_len + 2)
			xi_mid = mat(xi)
			xi_max = xi.max()
			xi_min = xi.min()
			seg_data = data[xi_min-1 : xi_max , 0]
			nrows_seg = shape(seg_data)[0]
			mid_trend = (A_coeff * seg_data).T

		#right trend

			if (seg_index + 1) * (seg_len - 1) + 1 > data_len :
				xi = np.arange(seg_index * (seg_len - 1) + 1 , data_len + 1)
				xi_right = mat(xi)
				xi_max = xi.max()
				xi_min = xi.min()
				seg_data = data[xi_min - 1 : xi_max , 0]
				nrows_seg = shape(seg_data)[0]

				if nrows_seg < seg_len :
					coeff_output1 , A1 = dc.detrending_coeff(nrows_seg , fit_order)
					A_coeff1 = A1 * coeff_output1
					right_trend = (A_coeff1 * seg_data).T
				else :
					right_trend = (A_coeff * seg_data).T

				xx1 = left_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
				xx2 = mid_trend[0 , 0 : int((seg_len + 1) / 2)]
				w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
				xx_left = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

				xx1 = mid_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
				xx2 = right_trend[0 , 0 : int((seg_len + 1) / 2)]
				w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
				xx_right = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

				record_x = xi_left[0 , 0 : nonoverlap_len]
				record_y = left_trend[0 , 0 : nonoverlap_len]

				record_x = np.hstack((record_x , xi_left[0 , int((shape(xi_left)[1] + 1) / 2) - 1 : shape(xi_left)[1]] , xi_mid[0 , int((shape(xi_mid)[1] + 1) / 2) : shape(xi_mid)[1]]))
				record_y = hstack((record_y , xx_left[0 , 0 : shape(xx_left)[1]] , xx_right[0 , 1 : shape(xx_right)[1]]))

				right_start_index = mat([(j) for j in range(shape(xi_right)[1]) if xi_right[0 , j] == xi_mid[0 , shape(xi_mid)[1] - 1] + 1])
				right_start_index =right_start_index[0 , 0]
				record_x = hstack((record_x,xi_right[0 , right_start_index : shape(xi_right)[1]]))
				record_y = hstack((record_y,right_trend[0 , right_start_index : shape(right_trend)[1]]))
				detrended_data = data - record_y.T

				return  detrended_data , record_y

			else :
				xi = np.arange(seg_index * (seg_len - 1) + 1 , (seg_index + 1) * (seg_len - 1) + 2)
				xi_right = mat(xi)
				xi_max = xi.max()
				xi_min = xi.min()
				seg_data = data[xi_min - 1 : xi_max,0]
				right_trend = (A * coeff_output * seg_data).T

				xx1 = left_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
				xx2 = mid_trend[0 , 0 : int((seg_len + 1) / 2)]
				w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
				xx_left = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

				xx1 = mid_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
				xx2 = right_trend[0 , 0 : int((seg_len + 1) / 2)]
				w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
				xx_right = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

				record_x = xi_left[0 , 0 : nonoverlap_len]
				record_y = left_trend[0 , 0 : nonoverlap_len]

				record_x = hstack((record_x , xi_left[0 , int((shape(xi_left)[1] + 1) / 2) - 1 : shape(xi_left)[1]] , xi_mid[0 , int((shape(xi_mid)[1] + 1) /2 ) : shape(xi_mid)[1]]))
				record_y = hstack((record_y , xx_left[0 , 0 : shape(xx_left)[1]] , xx_right[0 , 1 : shape(xx_right)[1]]))


	for seg_index in range(2 , int((data_len - 1) / (seg_len - 1))) :
		#left_trend
		#seg_index = 1
		xi = np.arange((seg_index - 1) * (seg_len - 1) + 1 , seg_index * (seg_len - 1) + 2)
		xi_left = mat(xi)
		xi_max = xi.max()
		xi_min = xi.min()
		seg_data = data[xi_min - 1 : xi_max , 0]
		left_trend = (A_coeff * seg_data).T

		# mid trend

		xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) + nonoverlap_len , seg_index * (seg_len -1) + nonoverlap_len + 2)
		xi_mid = mat(xi)
		xi_max = xi.max()
		xi_min = xi.min()
		seg_data = data[xi_min - 1 : xi_max , 0]
		mid_trend = (A_coeff * seg_data).T

		# right trend

		xi = np.arange(seg_index * (seg_len - 1) + 1 , (seg_index + 1) * (seg_len - 1) + 2)
		xi_right = mat(xi)
		xi_max = xi.max()
		xi_min = xi.min()
		seg_data = data[xi_min - 1 : xi_max , 0]
		right_trend = (A_coeff * seg_data).T

		xx1 = left_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
		xx2 = mid_trend[0 , 0 : int((seg_len + 1) / 2)]
		w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
		xx_left = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

		xx1 = mid_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
		xx2 = right_trend[0 , 0 : int((seg_len + 1) / 2)]
		w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
		xx_right = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

		record_x = hstack((record_x , xi_left[0 , int((shape(xi_left)[1] + 3) / 2) - 1 : shape(xi_left)[1]] , xi_mid[0 , int((shape(xi_mid)[1] + 1) / 2) : shape(xi_mid)[1]]))
		record_y = hstack((record_y , xx_left[0 , 1 : shape(xx_left)[1]] , xx_right[0 , 1 : shape(xx_right)[1]]))

#last part of data

	for seg_index in range(int((data_len - 1) / (seg_len - 1)) , int((data_len - 1) / (seg_len - 1)) + 1) :
	# left trend
	#seg_index = 1

		xi = np.arange((seg_index - 1) * (seg_len - 1) + 1 , seg_index * (seg_len - 1) + 2)
		xi_left = mat(xi)
		xi_max = xi.max()
		xi_min = xi.min()
		seg_data = data[xi_min - 1 : xi_max , 0]
		left_trend = (A_coeff * seg_data).T

		# mid trend

		if seg_index * (seg_len - 1) + 1 + nonoverlap_len > data_len :
			xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) + nonoverlap_len , data_len+ 1)
			xi_mid = mat(xi)
			xi_max = xi.max()
			xi_min = xi.min()
			seg_data = data[xi_min - 1 : xi_max , 0]
			nrows_seg = shape(seg_data)[0]

			if nrows_seg < seg_len :
				coeff_output1 , A1 = dc.detrending_coeff(nrows_seg , fit_order)
				A_coeff1 = A1 * coeff_output1
				mid_trend = (A_coeff1 * seg_data).T
			else :
				mid_trend = (A_coeff * seg_data).T

			xx1 = left_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
			xx2  =mid_trend[0 , 0 : int((seg_len + 1) / 2)]
			w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
			xx_left = multiply(xx1 , (1 - w)) + multiply(xx2 , w )
			mid_start_index = mat([(j) for j in range(shape(xi_mid)[1]) if xi_mid[0 , j] == xi_left[0 , shape(xi_left)[1] - 1] + 1])
			nrows_mid = shape(mid_start_index)[0]
			mid_start_index = mid_start_index[0 , 0]

			if nrows_mid == 0 :

				record_x = hstack((record_x , xi_left[0 , int((shape(xi_left)[1] + 3) / 2) - 1 : shape(xi_left)[1]]))
				record_y = hstack((record_y , xx_left[0 , 1 : shape(xx_left)[1]]))

			else :
				record_x = hstack((record_x , xi_left[0 , int((shape(xi_left)[1] + 3) / 2) - 1 : shape(xi_left)[1]] , xi_mid[0 , mid_start_index : shape(xi_mid)[1]]))
				record_y = hstack((record_y , xx_left[0 , 1 : shape(xx_left)[1]] , mid_trend[0 , int((seg_len + 3) / 2) - 1 : shape(mid_trend)[1]]))

			detrended_data = data - record_y.T


			return detrended_data , record_y

		else :
			xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) + nonoverlap_len , seg_index * (seg_len - 1) + nonoverlap_len + 2)
			xi_mid = mat(xi)
			xi_max = xi.max()
			xi_min = xi.min()
			seg_data = data[xi_min - 1 : xi_max , 0]
			mid_trend = (A_coeff * seg_data).T

		# right trend
		xi = np.arange(seg_index * (seg_len - 1) + 1 , data_len + 1)
		xi_right = mat(xi)
		xi_max = xi.max()
		xi_min = xi.min()
		seg_data = data[xi_min - 1 : xi_max , 0]
		nrows_seg = shape(seg_data)[0]

		if nrows_seg < seg_len :
			coeff_output1 , A1 = dc.detrending_coeff(nrows_seg , fit_order)
			A_coeff1 = A1 * coeff_output1
			right_trend = (A_coeff1 * seg_data).T
		else:
			right_trend = (A_coeff * seg_data).T

		xx1 = left_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
		xx2  =mid_trend[0 , 0 : int((seg_len + 1) / 2)]
		w = np.arange(0 , nonoverlap_len + 1)/nonoverlap_len
		xx_left = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

		xx1 = mid_trend[0 , int((seg_len + 1) / 2) - 1 : seg_len]
		xx2 = right_trend[0 , 0 : int((seg_len + 1) / 2)]
		w = np.arange(0 , nonoverlap_len + 1) / nonoverlap_len
		xx_right = multiply(xx1 , (1 - w)) + multiply(xx2 , w )

		record_x = hstack((record_x , xi_left[0 , int((shape(xi_left)[1] + 3) / 2) - 1 : shape(xi_left)[1]] , xi_mid[0 , int((shape(xi_mid)[1] + 1) / 2) : shape(xi_mid)[1]]))
		record_y = hstack((record_y , xx_left[0 , 1 : shape(xx_left)[1]] , xx_right[0 , 1 : shape(xx_right)[1]]))

		right_start_index = mat([(j) for j in range(shape(xi_right)[1]) if xi_right[0 , j] == xi_mid[0 , shape(xi_mid)[1] - 1] + 1])
		nrows_mid = shape(right_start_index)[1]

		if nrows_mid == 1 :
			right_start_index = right_start_index[0,0]
			record_x = hstack((record_x , xi_right[0 , right_start_index : shape(xi_right)[1]]))
			record_y = hstack((record_y , right_trend[0 , right_start_index : shape(right_trend)[1]]))

		detrended_data = data - record_y.T

		return detrended_data , record_y
