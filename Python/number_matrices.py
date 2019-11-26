from __future__ import division
import numpy as np
import parevol_tools as pt
import pandas as pd
from functools import reduce
import operator
import math
from decimal import Decimal

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def number_matrices(matrix_array):
    N_2 = np.sum(matrix_array == 2)
    N_3 = np.sum(matrix_array == 3)
    N_greater_3 = np.sum(matrix_array > 3)

    def array_permutations(sum_array, k):
        return sum([  prod( list(range(int(sum_i)-k+1, int(sum_i)+1)) ) for sum_i in sum_array])


    row_sums = matrix_array.sum(axis=1)
    m = len(row_sums)
    column_sums = matrix_array.sum(axis=0)
    n = len(column_sums)
    S = int(sum(row_sums))
    # s for rows, t for columns
    S_2 = array_permutations(row_sums, 2)
    T_2 = array_permutations(column_sums, 2)
    S_3 = array_permutations(row_sums, 3)
    T_3 = array_permutations(column_sums, 3)

    mu_hat_2 = ( (m*n) / (S*((m*n)+S)) ) * sum( [ (sum_i - (S/m))**2  for sum_i in row_sums ]  )
    nu_hat_2 = ( (m*n) / (S*((m*n)+S)) ) * sum( [ (sum_i - (S/n))**2  for sum_i in column_sums ]  )
    mu_hat_3 = ( (m*n) / (S*((m*n)+S)) ) * sum( [ (sum_i - (S/m))**3  for sum_i in row_sums ]  )
    nu_hat_3 = ( (m*n) / (S*((m*n)+S)) ) * sum( [ (sum_i - (S/n))**3  for sum_i in column_sums ]  )



    N_3_max = max( [ math.ceil(np.log(S)),  math.ceil(230000*S_3*T_3/(S**3)) ] )

    #def calculate_number_matrices:
        # using Corollary 4.1 with Sterling approximation


    if N_greater_3 > 0:
        raise RuntimeError('Matrix contains values greater than 3')

    elif N_3 > N_3_max:
        raise RuntimeError('Number of entries equal to 3 is greater than maximum')

    elif (S_2*T_2 < S ** (7/4)) and N_2>22:
        raise RuntimeError('Number of entries equal to 2 is greater than maximum')

    elif ((S ** (7/4)) <= S_2*T_2 < (1/5600)*(S**2)*np.log(S)) and (N_2 > math.ceil(np.log(S))):
        raise RuntimeError('Number of entries equal to 2 is greater than maximum')

    elif ((1/5600)*(S**2)*np.log(S) <= S_2*T_2) and (N_2 > 5600*S_2*T_2/(S**2)):
        raise RuntimeError('Number of entries equal to 2 is greater than maximum')


    else:
        # convert second term to int, shoudn't lose much precision
        comb_rows = ((n ** S) // prod( [math.factorial(row_sum) for row_sum in row_sums])) * \
                    int(math.exp( (S_2/(2*n)) - (S_2/(4*(n**2))) -(S_3/(6*(n**2))) ))
        comb_cols = ((m ** S) // prod( [math.factorial(col_sum) for col_sum in column_sums])) * \
                    int(math.exp( (T_2/(2*m)) - (T_2/(4*(m**2))) -(T_3/(6*(m**2))) ))

        combs = (((m*n)**S) // math.factorial(S)) * \
                int(math.exp( ((S**2)/(2*m*n)) - (S/(2*m*n)) - ((S**3)/(6*(m**2)*(n**2)))  ))

        M_term1 = (comb_rows*comb_cols) // combs
        M_term1_exp1 = ((1-mu_hat_2)*(1-nu_hat_2)) * (0.5 + ((3- (mu_hat_2*nu_hat_2)) / (4*S)))
        M_term1_exp2 = ((1-mu_hat_2)*(3+mu_hat_2 - (2*mu_hat_2*nu_hat_2)) ) / (4*n)
        M_term1_exp3 = ((1-nu_hat_2)*(3+nu_hat_2 - (2*mu_hat_2*nu_hat_2)) ) / (4*m)
        M_term1_exp4 = ((1- (3*(mu_hat_2**2)) + (2*mu_hat_3) ) * (1- (3*(nu_hat_2**2)) + (2*nu_hat_3)) ) / (12*S)
        M = M_term1 * int(math.exp( M_term1_exp1 - M_term1_exp2 - M_term1_exp3 +M_term1_exp4))

        print(M)
        print(str(M)[0] + '.' + str(M)[1:3] + '*10**' + str(len(str(M))-1) )



#df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
#df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)


number_matrices( df.values)

#number_matrices( np.asarray([[2,0],[1,1],[3,2]] ))
