import argparse
import glob
import os

from satbench.utils.utils import parse_cnf_file, clean_clauses, write_dimacs_to
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory with sat data')
    opts = parser.parse_args()
    print(opts)

    print('Cleaning formulas...')

    all_files = sorted(glob.glob(opts.data_dir + '/**/*.cnf', recursive=True))
    all_files = [os.path.abspath(f) for f in all_files if 'augmented' not in f]

    print(f'There are {len(all_files)} files.')

    for f in tqdm(all_files):
        n_vars, clauses = parse_cnf_file(f)
        clauses = clean_clauses(clauses)
        write_dimacs_to(n_vars, clauses, f)


if __name__ == '__main__':
    main()
