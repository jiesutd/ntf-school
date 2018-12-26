# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2018-12-26 09:17:26
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-12-26 14:55:57
from ncp import *
import numpy as np
from sktensor import ktensor
from sktensor.dtensor import dtensor
import time
import random
seed_num = 42
random.seed(seed_num)
np.random.seed(seed_num)



def main(input_numpy_array, decomp_rank = 10):
	print("Input array shape:", input_numpy_array.shape)
	print("Decompositation rank: ", decomp_rank)
	start_time = time.time()
	input_tensor = dtensor(input_numpy_array)
	X_approx_ks = nonnegative_tensor_factorization(input_tensor, decomp_rank, tol=1e-6, max_iter=300)
	X_err = (input_tensor - X_approx_ks.totensor()).norm()/input_tensor.norm()
	end_time = time.time()
	time_cost = end_time - start_time
	# print input_numpy_array
	print(X_approx_ks.U[0].shape) 
	print( X_approx_ks.U[1].shape )
	print( X_approx_ks.U[2].shape )
	print( X_err)
	print( "%.2fs"%time_cost)



if __name__ == '__main__':
	input_array = np.random.randint(15,size=(50,60,70))
	decomp_rank = 10
	main(input_array, decomp_rank)