import os
import argparse
import glob
import pickle
import subprocess
import numpy as np
import networkx as nx

from concurrent.futures.process import ProcessPoolExecutor
from g4satbench.utils.utils import ROOT_DIR, parse_cnf_file, parse_proof_file, hash_clauses, write_dimacs_to


class Generator:
    def __init__(self, split):
        self.split = split
        self.exec_dir = os.path.join(ROOT_DIR, 'external')

    def run(self, cnf_filepath):
        filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
        proof_filepath = os.path.join(os.path.dirname(cnf_filepath), filename + '.proof')

        if 1:
            prover_cmd_line = ['./cadical', '--no-binary', cnf_filepath, proof_filepath]

            try:
                process = subprocess.Popen(prover_cmd_line, cwd=self.exec_dir, start_new_session=True)
                process.communicate()
            except:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        if not os.path.exists(proof_filepath):
            return

        n_vars, clauses = parse_cnf_file(cnf_filepath)
        
        learned_clauses, deleted_clauses = parse_proof_file(proof_filepath)

        formula_filepath = os.path.join(os.path.dirname(os.path.dirname(cnf_filepath)), f'augmented_{self.split}/' + filename + '.cnf')
        write_dimacs_to(n_vars, clauses, formula_filepath, learned_clauses)

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--splits', type=str, nargs='+')
    parser.add_argument('--n_process', type=int, default=10, help='Number of processes to run')

    opts = parser.parse_args()

    for split in opts.splits:
        os.makedirs(os.path.join(opts.input_dir, f'augmented_{split}'), exist_ok=True)

        generator = Generator(split)

        all_files = sorted(glob.glob(opts.input_dir + f'/{split}/*.cnf', recursive=True))
        assert len(all_files) > 0
        all_files = [os.path.abspath(f) for f in all_files]
        
        with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
            pool.map(generator.run, all_files)


if __name__ == '__main__':
    main()
