import argparse, os
import run_simulations as rs

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

    if analysis == True:
        run_ba_cov_sims_N_out = out_path + '/data/simulations/' + 'ba_cov_N_sims' + '.txt'
        rs.run_ba_cov_sims(gene_list=[50], pop_list=[2, 4, 8, 16, 32, 64],
                out_name = run_ba_cov_sims_N_out, iter1=10000, iter2=10000)
        #run_ba_cov_sims_G_out = out_path + '/data/simulations/' + 'ba_cov_G_sims' + '.txt'
        #run_ba_cov_sims(gene_list=[8, 16, 32, 64, 128], pop_list=[50],
        #        out_name = run_ba_cov_sims_G_out, iter1=10000, iter2=10000)
        #run_ba_cov_neutral_sims_out = out_path + '/data/simulations/' + 'ba_cov_neutral_sims' + '.txt'
        #run_ba_cov_neutral_sims(run_ba_cov_neutral_sims_out, shape=1, scale=1,
        #    G = 50, N = 50, iter1=10000, iter2=10000)
