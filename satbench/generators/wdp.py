import os
import argparse
import random
import subprocess
import networkx as nx

from pysat.solvers import Cadical
from satbench.utils.utils import ROOT_DIR, parse_cnf_file, write_dimacs_to, VIG, clean_clauses, hash_clauses
from tqdm import tqdm


class Generator:
    def __init__(self, opts):
        self.opts = opts
        self.hash_list = []
        self.exec_dir = os.path.join(ROOT_DIR, 'external')

    def run(self):
        for split in ['train', 'valid', 'test']:
            n_instances = getattr(self.opts, f'{split}_instances')
            if n_instances > 0:
                sat_out_dir = os.path.join(os.path.abspath(self.opts.out_dir), f'{split}/sat')
                os.makedirs(sat_out_dir, exist_ok=True)
                print(f'Generating ca {split} set...')
                for i in tqdm(range(n_instances)):
                    self.generate(i, sat_out_dir)

    def generate(self, i, sat_out_dir):
        while True:
            # Number of factorys
            f = random.randint(self.opts.min_f, self.opts.max_f)
            # Number of workers
            w = random.randint(self.opts.min_w, self.opts.max_w)
            # Number of jobs
            j = random.randint(self.opts.min_j, self.opts.max_j)
            # The probability of a worker can do a job
            p = random.uniform(self.opts.min_p, self.opts.max_p)
            
            table_filepath = os.path.abspath(os.path.join(self.opts.out_dir, '%.5d.txt' % (i)))
            cmd_line = ['./wdp2table', str(f), str(w), str(j), str(p)]
            with open(table_filepath, 'w') as f_out:
                try:
                    process = subprocess.Popen(cmd_line, stdout=f_out, stderr=f_out, cwd=self.exec_dir, start_new_session=True)
                    process.communicate()
                except:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            if os.stat(table_filepath).st_size == 0:
                os.remove(table_filepath)
                continue

            cnf_filepath = os.path.abspath(os.path.join(self.opts.out_dir, '%.5d.cnf' % (i)))
            cmd_line = ['./table2cnf', str(f), str(w), str(j), table_filepath]

            with open(cnf_filepath, 'w') as f_out:
                try:
                    process = subprocess.Popen(cmd_line, stdout=f_out, stderr=f_out, cwd=self.exec_dir, start_new_session=True)
                    process.communicate()
                except:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            if os.stat(cnf_filepath).st_size == 0:
                os.remove(cnf_filepath)
                continue
            
            n_vars, clauses = parse_cnf_file(cnf_filepath)
            vig = VIG(n_vars, clauses)
            if not nx.is_connected(vig):
                os.remove(table_filepath)
                os.remove(cnf_filepath)
                continue
            
            clauses = clean_clauses(clauses)
            h = hash_clauses(clauses)

            if h in self.hash_list:
                continue

            solver = Cadical(bootstrap_with=clauses)

            if solver.solve():
                self.hash_list.append(h)
                write_dimacs_to(n_vars, clauses, os.path.join(sat_out_dir, '%.5d.cnf' % (i)))
                os.remove(table_filepath)
                os.remove(cnf_filepath)
                break
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)

    parser.add_argument('--train_instances', type=int, default=0)
    parser.add_argument('--valid_instances', type=int, default=0)
    parser.add_argument('--test_instances', type=int, default=0)

    parser.add_argument('--min_f', type=int, default=15)
    parser.add_argument('--max_f', type=int, default=25)

    parser.add_argument('--min_w', type=int, default=35)
    parser.add_argument('--max_w', type=int, default=45)

    parser.add_argument('--min_j', type=int, default=15)
    parser.add_argument('--max_j', type=int, default=25)

    parser.add_argument('--min_p', type=float, default=0.85)
    parser.add_argument('--max_p', type=float, default=0.95)

    parser.add_argument('--seed', type=int, default=0)

    opts = parser.parse_args()

    random.seed(opts.seed)

    generator = Generator(opts)
    generator.run()


if __name__ == '__main__':
    main()
