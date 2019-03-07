import argparse, os
import run_simulations as rs
import numpy as np

### This file will contain all the commands to automatically rerun all analyses.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Run analyses and/or generate figures.")
    parser.add_argument('-a', '--analysis', default = False, action='store_true',\
        help = 'Command to run analyses')
    parser.add_argument('-f', type = str, default = '0', help = "Command to generate figures")
    parser.add_argument('-c', '--carbonate', default = False, action='store_true',\
        help = 'Command to run analyses')

    args = parser.parse_args()
    analysis = args.analysis
    carbonate = args.carbonate
    figure = args.f.upper()

    if carbonate == True:
        out_path = '/N/dc2/projects/Lennon_Sequences/ParEvol'
    else:
        out_path = os.path.expanduser("~/GitHub/ParEvol")

    iter1=1000
    iter2=1000

    if analysis == True:
        #N_out = out_path + '/data/simulations/' + 'ba_cov_N_sims' + '.txt'
        #rs.run_ba_cov_sims(gene_list=[50], pop_list=[2, 4, 8, 16, 32, 64],
        #        out_name = N_out, covs = [0.1, 0.15, 0.2], iter1=iter1, iter2=iter2)

        #G_out = out_path + '/data/simulations/' + 'ba_cov_G_sims' + '.txt'
        #run_ba_cov_sims(gene_list=[8, 16, 32, 64, 128], pop_list=[50],
        #        out_name = G_out, covs = [0.1, 0.15, 0.2], iter1=iter1, iter2=iter2)

        #neuts_out = out_path + '/data/simulations/' + 'ba_cov_neutral_sims' + '.txt'
        #rs.run_ba_cov_neutral_sims(neuts_out, covs = [0.1, 0.15, 0.2],
        #    shape=1, scale=1, G = 50, N = 50, iter1=iter1, iter2=iter2)

        #props_out = out_path + '/data/simulations/' + 'ba_cov_prop_sims' + '.txt'
        #props = np.linspace(0, 1, num = 20)
        #rs.run_ba_cov_prop_sims(props_out, covs = [0.1, 0.15, 0.2],
        #    props=props, shape=1, scale=1, G = 50, N = 50, iter1=iter1, iter2=iter2)

        # rho vs power
        #rhos_out = out_path + '/data/simulations/' + 'ba_cov_rho_sims.txt'
        #rs.run_ba_cov_rho_sims(rhos_out, covs = [0.2, 0.15, 0.1], rhos=[-0.3, -0.15, 0, 0.15, 0.3],
        #    shape=1, scale=1, G = 50, N = 50, iter1=iter1, iter2=iter2)

        '''two treatment sims'''
        # distance vs. power
        dist_out = out_path + '/data/simulations/' + 'ba_cov_dist_sims' + '.txt'
        rs.run_cov_dist_sims(dist_out, to_reshuffle =[1,2,3,4,5], N1=20, N2=20, G=100,
            covs = [0.1, 0.15, 0.2], shape = 1, scale = 1, iter1=10, iter2=1000))
