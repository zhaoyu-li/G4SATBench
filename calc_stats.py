import argparse
import glob
import os
import networkx as nx
import networkx.algorithms.community as nx_comm
import glob
import random

from tqdm import tqdm
from satbench.utils.utils import parse_cnf_file, VIG, VCG, LCG, clean_clauses
from collections import defaultdict


terms = ['n_vars', 'n_clauses', 'vig-clustering_coefficient', 'vig-modularity', 'vcg-modularity', 'lcg-modularity']

def calc_stats(f):
    n_vars, clauses = parse_cnf_file(f)
    clauses = clean_clauses(clauses)
    vig = VIG(n_vars, clauses)
    vcg = VCG(n_vars, clauses)
    lcg = LCG(n_vars, clauses)

    return {
        'n_vars': n_vars,
        'n_clauses': len(clauses),
        'vig-clustering_coefficient': nx.average_clustering(vig),
        'vig-modularity': nx_comm.modularity(vig, nx_comm.louvain_communities(vig)),
        'vcg-modularity': nx_comm.modularity(vcg, nx_comm.louvain_communities(vcg)),
        'lcg-modularity': nx_comm.modularity(lcg, nx_comm.louvain_communities(lcg)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory with sat data')
    opts = parser.parse_args()
    print(opts)

    print('Calculating statistics...')

    all_files = sorted(glob.glob(opts.data_dir + '/**/*.cnf', recursive=True))
    all_files = [os.path.abspath(f) for f in all_files if 'augmented' not in f]

    stats = defaultdict(int)

    for f in tqdm(all_files):
        s = calc_stats(f)
        for t in terms:
            stats[t] += s[t]

    for t in terms:
        print('%30s\t%10.2f' % (t, stats[t] / len(all_files)))


if __name__ == '__main__':
    main()
