import os
import argparse
import glob
import pickle
import subprocess
import numpy as np
import itertools

from concurrent.futures.process import ProcessPoolExecutor
from ..utils.utils import ROOT_DIR


class Generator:
    def __init__(self):
        self.exec_dir = os.path.join(ROOT_DIR, 'external')

    def run(self, cnf_filepath):
        filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
        proof_filepath = os.path.join(os.path.dirname(cnf_filepath), filename + '.proof')
        core_filepath = os.path.join(os.path.dirname(cnf_filepath), filename + '.core')

        if not os.path.exists(proof_filepath):
            prover_cmd_line = ['./cadical', '--unsat', cnf_filepath, proof_filepath]

            try:
                process = subprocess.Popen(prover_cmd_line, cwd=self.exec_dir, start_new_session=True)
                process.communicate()
            except:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        if not os.path.exists(proof_filepath):
            return
        
        if not os.path.exists(core_filepath):
            checker_cmd_line = ['./drat-trim', cnf_filepath, proof_filepath, '-c', core_filepath]

            try:
                process = subprocess.Popen(checker_cmd_line, cwd=self.exec_dir, start_new_session=True)
                process.communicate()
            except:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        if not os.path.exists(core_filepath):
            return
        
        with open(core_filepath, 'r') as f:
            lines = f.readlines()
            header = lines[0].strip().split()
            n_vars = int(header[2])
            n_clauses = int(header[3])

            core_variable = np.zeros(n_vars)
            for line in lines[1:]:
                tokens = line.strip().split()
                core_variable[[abs(int(t))-1 for t in tokens[:-1]]] = 1
        
        core_variable_file = os.path.join(os.path.dirname(cnf_filepath), filename + '_core_variable.pkl')
        
        with open(core_variable_file, 'wb') as f:
            pickle.dump(core_variable, f)

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('splits', type=str, nargs='+')
    parser.add_argument('--n_process', type=int, default=10, help='Number of processes to run')

    opts = parser.parse_args()

    generator = Generator()

    all_files = [sorted(glob.glob(opts.input_dir + f'/{split}/*.cnf', recursive=True)) for split in opts.splits]
    all_files = list(itertools.chain(*all_files))

    assert len(all_files) > 0
    all_files = [os.path.abspath(f) for f in all_files]

    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        pool.map(generator.run, all_files)




if __name__ == '__main__':
    main()
