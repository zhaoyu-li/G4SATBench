import os
import argparse
import glob
import pickle
import subprocess
import numpy as np

from concurrent.futures.process import ROOT_DIR, ProcessPoolExecutor


class Generator:
    def __init__(self):
        self.exec_dir = os.path.join(ROOT_DIR, 'external')
        
    def run(self, cnf_filepath):
        filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
        proof_filepath = os.path.join(os.path.dirname(cnf_filepath), filename + '.proof')

        if not os.path.exists(proof_filepath):
            prover_cmd_line = ['./cadical', '--unsat', cnf_filepath, proof_filepath]
            try:
                process = subprocess.Popen(prover_cmd_line, cwd=self.exec_dir, start_new_session=True)
                process.communicate()
            except:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        if not os.path.exists(proof_filepath):
            return

        formula_filepath = os.path.join(os.path.dirname(os.path.dirname(cnf_filepath)), 'trimmed/' + filename + '.cnf')
        checker_cmd_line = ['./drat-trim', cnf_filepath, proof_filepath, '-c', formula_filepath]

        try:
            process = subprocess.Popen(checker_cmd_line, cwd=self.exec_dir, start_new_session=True)
            process.communicate()
        except:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--n_process', type=int, default=10, help='Number of processes to run')

    opts = parser.parse_args()

    os.makedirs(os.path.join(opts.input_dir, 'trimmed'), exist_ok=True)

    generator = Generator()

    all_files = sorted(glob.glob(opts.input_dir + '/unsat/*.cnf', recursive=True))
    assert len(all_files) > 0
    all_files = [os.path.abspath(f) for f in all_files]
    
    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        pool.map(generator.run, all_files)
    

if __name__ == '__main__':
    main()
