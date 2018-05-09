from __future__ import division
import numpy as np


array = np.array([[ 1,  0.],
   [ 0.,  0.],
   [ 1.,  2.],
   [ 3.,  1.],
   [ 0.,  1.]])


def sample_matrix(array):
    row_sum = array.sum(axis=1)
    column_sum = array.sum(axis=0)
    M = sum(row_sum)
    sample_array = array
    # sample over columns
    for j, column in enumerate(array.T):
        a_j = []
        for i, a_ij in enumerate(column):
            if (i == 0):
                low = max(0, column_sum[j] + row_sum[i] - M)
                high = min(column_sum[j], row_sum[i])
                # array[row, column]
                #sample_array[0, j] = np.random.randint(low, high = high + 1)
            else:
                # for 0 <= i <= k-1
                # sample a[j+1, i]
                sum_a_i_1 = sum(sample_array[0:i, j])
                low = max(0, (column_sum[j] - sum_a_i_1) -  sum(row_sum[i+1:]))
                high = min(row_sum[i], (column_sum[j] - sum_a_i_1) )
            # low = inclusive, high = exclusive
            sample_array[i, j] = np.random.randint(low, high = high + 1)

    print(sample_array)
    return sample_array



sample_matrix(array)
