#import rpy2.robjects as robjects

# function to generate confidence intervals based on Fisher Information criteria
def CI_FIC(results):
    # standard errors = square root of the diagnol of a variance-covariance matrix
    ses = np.sqrt(np.absolute(np.diagonal(results.cov_params())))
    cfs = results.params
    lw = cfs - (1.96*ses)
    up = cfs +(1.96*ses)
    return (lw, up)


def get_broken_stick(array):
    # Legendre & Legendre, eqn. 9.16
    array = np.sort(array)
    out_list = []
    for j in range(1, len(array)+1):
        #print(sum((1/x) for x in range(j, len(array)) ))
        out_list.append(sum((1/x) for x in range(j, len(array)+1) ))
    return np.asarray(out_list) * (1 / len(array))




def ellipse_polyline(ellipses, n=100):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    st = np.sin(t)
    ct = np.cos(t)
    result = []
    for x0, y0, a, b, angle in ellipses:
        angle = np.deg2rad(angle)
        sa = np.sin(angle)
        ca = np.cos(angle)
        p = np.empty((n, 2))
        p[:, 0] = x0 + a * ca * ct - b * sa * st
        p[:, 1] = y0 + a * sa * ct + b * ca * st
        result.append(p)
    return result




def get_theta_from_cov(C):
    eVa, eVe = np.linalg.eig(C)
    var_1 = C[0,0]
    var_2 = C[1,1]
    if C[0,1] > 0.0:
        if abs(round(math.degrees(math.acos(eVe[0,0])), 3)) > 90:
            theta = (180 - abs(round(math.degrees(math.acos(eVe[0,0])), 3)))
        else:
            theta = abs(round(math.degrees(math.acos(eVe[0,0])), 3))

    elif C[0,1] < 0.0:
        if abs(round(math.degrees(math.acos(eVe[0,0])), 3)) > 90:
            theta = -(180 - abs(round(math.degrees(math.acos(eVe[0,0])), 3)))
        else:
            theta = -abs(round(math.degrees(math.acos(eVe[0,0])), 3))
    else:
        theta = 0
    major_axis_length = 2 * math.sqrt(5.991 * eVa[0])
    minor_axis_length = 2 * math.sqrt(5.991 * eVa[1])
    return major_axis_length, minor_axis_length, theta



def get_mean_angle(array, k = 3):
    def angle_between(v1, v2):
        radians = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return radians * 180 / math.pi

    X = array[:,0:k]
    rows = list(range(array.shape[0]))
    angle_pairs = []
    for i in rows:
        for j in rows:
            if i < j:
                row_i = X[i,:]
                row_j = X[j,:]
                angle_pairs.append( angle_between(row_i, row_j) )

    return (sum(angle_pairs) * 2) / (len(rows) * (len(rows)-1))


def get_euc_magnitude_diff(array, k = 3):
    X = array[:,0:k]
    rows = list(range(array.shape[0]))
    angle_pairs = []
    for i in rows:
        for j in rows:
            if i < j:
                row_i = X[i,:]
                row_j = X[j,:]
                # difference in magnitude
                angle_pairs.append( abs(np.linalg.norm(row_i) - np.linalg.norm(row_j)) )

    return (sum(angle_pairs) * 2) / (len(rows) * (len(rows)-1))



def get_pcoa(df):
    # remove columns containing only zeros
    df_no0 = df.loc[:, (df != 0).any(axis=0)]
    # only keep pops from day 100
    ids = df_no0.index.values
    data = df_no0.values
    ds = get_ds(data)
    pcoa = cmdscale(ds)
    Y = pd.DataFrame(pcoa[0])
    Y['pops'] = ids
    Y = Y.set_index('pops')
    return([Y, pcoa[1]])


def cmdscale(D):
    """
    Classical multidimensional scaling (MDS)

    Parameters
    ----------
    D : (n, n) array
        Symmetric distance matrix.

    Returns
    -------
    Y : (n, p) array
        Configuration matrix. Each column represents a dimension. Only the
        p dimensions corresponding to positive eigenvalues of B are returned.
        Note that each dimension is only determined up to an overall sign,
        corresponding to a reflection.

    e : (n,) array
        Eigenvalues of B.
    Acquired from http://www.nervouscomputer.com/hfs/cmdscale-in-python/
    """
    # Number of points
    n = len(D)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n))/n

    # YY^T
    B = -H.dot(D**2).dot(H)/2

    print(B.T * B)

    # Diagonalize
    evals, evecs = np.linalg.eigh(B)

    # Sort by eigenvalue in descending order
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)

    return Y, evals


def get_scipy_bray_curtis(array):
    return squareform(pdist(array, metric = 'braycurtis'))



def partition(lst, n):
    # partitions a list into n lists of equal length
    random.shuffle(lst)
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]



def perm_unique(elements):
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)


def get_adjacency_matrix(array):
    array = np.transpose(array)
    adjacency_array = np.zeros((array.shape[0], array.shape[0]))
    for i, row_i in enumerate(array):
        for j, row_j in enumerate(array):
            if i <= j:
                continue
            test = [1 if ((x[0] > 0) and (x[1] > 0)) else 0 for x in list(zip(row_i, row_j))  ]
            if sum(test) > 0:
                adjacency_array[i,j] = adjacency_array[j,i] = 1
            else:
                adjacency_array[i,j] = adjacency_array[j,i] = 0

    return adjacency_array



'''code is from https://stackoverflow.com/questions/6284396/permutations-with-unique-values'''

class unique_element:
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences

def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1

#def random_matrix(array):
#    ### use
#    ###  switch to ASA159 algorithm
#    r2dtable = robjects.r['r2dtable']
#    row_sum = array.sum(axis=1)
#    column_sum = array.sum(axis=0)
#    sample = r2dtable(1, robjects.IntVector(row_sum), robjects.IntVector(column_sum))
#    return np.asarray(sample[0])


def run_pca_permutation(iter = 10000, analysis = 'PCA', dataset = 'tenaillon'):
    if dataset == 'tenaillon':
        k = 3
        df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        df_array = df.as_matrix()
        df_out = open(pt.get_path() + '/data/Tenaillon_et_al/permute_' + analysis + '.txt', 'w')
        column_headers = ['Iteration', 'MCD', 'mean_angle', 'mean_dist', 'delta_L', 'x_stat']
        df_out.write('\t'.join(column_headers) + '\n')
        for i in range(iter):
            print(i)
            df_rndm = pd.DataFrame(data=pt.random_matrix(df_array), index=df.index, columns=df.columns)
            df_rndm_delta = pt.likelihood_matrix(df_rndm, 'Tenaillon_et_al').get_likelihood_matrix()
            if analysis == 'PCA':
                X = pt.hellinger_transform(df_rndm_delta)
                pca = PCA()
                df_rndm_delta_out = pca.fit_transform(X)
                #df_pca = pd.DataFrame(data=X_pca, index=df.index)
            mean_angle = pt.get_mean_angle(df_rndm_delta_out, k = k)
            mcd = pt.get_mean_centroid_distance(df_rndm_delta_out, k=k)
            mean_length = pt.get_euc_magnitude_diff(df_rndm_delta_out, k=k)
            mean_dist = pt.get_mean_pairwise_euc_distance(df_rndm_delta_out, k=k)
            x_stat = pt.get_x_stat(pca.explained_variance_[:-1])
            df_out.write('\t'.join([str(i), str(mcd), str(mean_angle), str(mean_dist), str(mean_length), str(x_stat)]) + '\n')
        df_out.close()


    elif dataset == 'good':
        k = 5
        df_path = pt.get_path() + '/data/Good_et_al/gene_by_pop.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        to_exclude = pt.complete_nonmutator_lines()
        to_exclude.append('p5')
        df_nonmut = df[df.index.str.contains('|'.join( to_exclude))]
        # remove columns with all zeros
        df_nonmut = df_nonmut.loc[:, (df_nonmut != 0).any(axis=0)]
        time_points = [ int(x.split('_')[1]) for x in df_nonmut.index.values]
        time_points_set = sorted(list(set([ int(x.split('_')[1]) for x in df_nonmut.index.values])))
        df_nonmut_array = df_nonmut.as_matrix()
        time_points_positions = {}
        for x in time_points_set:
            time_points_positions[x] =  [i for i,j in enumerate(time_points) if j == x]
        df_final = df_nonmut.iloc[time_points_positions[time_points_set[-1]]]

        df_out = open(pt.get_path() + '/data/Good_et_al/permute_' + analysis + '.txt', 'w')
        #column_headers = ['Iteration', 'Generation', 'MCD']
        column_headers = ['Iteration', 'Generation', 'MCD', 'mean_angle', 'delta_L', 'mean_dist']
        df_out.write('\t'.join(column_headers) + '\n')
        for i in range(iter):
            print("Iteration " + str(i))
            matrix_0 = df_nonmut.iloc[time_points_positions[time_points_set[0]]]
            matrix_0_rndm = pt.random_matrix(matrix_0.as_matrix())
            df_rndm_list = [pd.DataFrame(data=matrix_0_rndm, index=matrix_0.index, columns=matrix_0.columns)]
            # skip first time step
            for j, tp in enumerate(time_points_set[0:]):
                if j == 0:
                    continue
                df_tp_minus1 = df_nonmut[df_nonmut.index.str.contains('_' + str(time_points_set[j-1]))]
                df_tp = df_nonmut[df_nonmut.index.str.contains('_' + str(tp))]
                matrix_diff = df_tp.as_matrix() - df_tp_minus1.as_matrix()
                matrix_0_rndm = matrix_0_rndm +  pt.random_matrix(matrix_diff)
                df_0_rndm = pd.DataFrame(data=matrix_0_rndm, index=df_tp.index, columns=df_tp.columns)
                df_rndm_list.append(df_0_rndm)

            df_rndm = pd.concat(df_rndm_list)
            df_rndm_delta = pt.likelihood_matrix(df_rndm, 'Good_et_al').get_likelihood_matrix()
            if analysis == 'PCA':
                X = pt.hellinger_transform(df_rndm_delta)
                pca = PCA()
                matrix_rndm_delta_out = pca.fit_transform(X)
            elif analysis == 'cMDS':
                matrix_rndm_delta_bc = np.sqrt(pt.get_bray_curtis(df_rndm_delta.as_matrix()))
                matrix_rndm_delta_out = pt.cmdscale(matrix_rndm_delta_bc)[0]
            else:
                print("Analysis argument not accepted")
                continue

            df_rndm_delta_out = pd.DataFrame(data=matrix_rndm_delta_out, index=df_rndm_delta.index)
            for tp in time_points_set:
                df_rndm_delta_out_tp = df_rndm_delta_out[df_rndm_delta_out.index.str.contains('_' + str(tp))]
                df_rndm_delta_out_tp_matrix = df_rndm_delta_out_tp.as_matrix()
                mean_angle = pt.get_mean_angle(df_rndm_delta_out_tp_matrix, k = k)
                mcd = pt.get_mean_centroid_distance(df_rndm_delta_out_tp_matrix, k=k)
                mean_length = pt.get_euc_magnitude_diff(df_rndm_delta_out_tp_matrix, k=k)
                mean_dist = pt.get_mean_pairwise_euc_distance(df_rndm_delta_out_tp_matrix, k=k)
                df_out.write('\t'.join([str(i), str(tp), str(mcd), str(mean_angle), str(mean_length), str(mean_dist) ]) + '\n')

        df_out.close()






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
