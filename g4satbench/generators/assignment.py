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
        model_filepath = os.path.join(os.path.dirname(cnf_filepath), filename + '.model')

        if not os.path.exists(model_filepath):
            cmd_line = ['./cadical', '--sat', cnf_filepath, '-w', model_filepath]

            try:
                process = subprocess.Popen(cmd_line, cwd=self.exec_dir, start_new_session=True)
                process.communicate()
            except:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        if not os.path.exists(model_filepath):
            return
            
        assignment = []
        with open(model_filepath, 'r') as f:
            lines = f.readlines()
            assert lines[0].strip().split()[1] == 'SATISFIABLE'

            for line in lines[1:]:
                assignment.extend([int(s) for s in line.strip().split()[1:]])
            
            assignment = np.array(assignment[:-1]) > 0 # ends with 0

        assignment_file = os.path.join(os.path.dirname(cnf_filepath), filename + '_assignment.pkl')
        
        with open(assignment_file, 'wb') as f:
            pickle.dump(assignment, f)

        
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
