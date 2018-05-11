
def sis_matrix(array):

    # sequential importance sampling (SIS) procedure for matrices with
    # fixed marginal sums to produce Monte Carlo samples close to the
    # uniform distribution
    # algoritm from Chen et al., 2005 doi: 10.1198/016214504000001303
    row_sum = array.sum(axis=1)
    column_sum = array.sum(axis=0)
    if sum(row_sum) != sum(column_sum):
        return "Error! Sum or row sums does not equal sum of column sums"
    M = sum(row_sum)
    sample_array = np.zeros((array.shape[0], array.shape[1]))
    # sample over columns
    for j, column in enumerate(array.T):
        a_j = []
        for i, a_ij in enumerate(column):
            if (i == 0):
                low = max(0, column_sum[j] + row_sum[i] - M)
                high = min(column_sum[j], row_sum[i])
            else:
                # for 0 <= i <= k-1
                # sample a[j+1, i]
                sum_a_i_1 = sum(sample_array[0:i, j])
                print(sample_array[0:i, j])
                print(sum_a_i_1)
                if i == len(row_sum) -1:
                    sum_row_sum_iPlus1_to_m = 0
                else:
                    sum_row_sum_iPlus1_to_m = sum(row_sum[i+1:])
                #print(sample_array[0:i, j])
                #print(sum_a_i_1)
                low = max(0, (column_sum[j] - sum_a_i_1) - sum_row_sum_iPlus1_to_m )
                #print(sample_array[0:i,j])
                #print(0, (column_sum[j] - sum_a_i_1) -  sum(row_sum[i+1:]))
                #print(column_sum[j], row_sum[i+1:])
                #print(column_sum[j] -row_sum[i+1:])
                #print(row_sum[i], (column_sum[j] - sum_a_i_1) )
                high = min(row_sum[i], (column_sum[j] - sum_a_i_1) )
            # low = inclusive, high = exclusive
            sample_array[i, j] = np.random.randint(low, high = high +1)

    return sample_array
