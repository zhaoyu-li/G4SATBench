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
    
    def random_binary_string(self, n):
        return ''.join([str(random.randint(0, 1)) for _ in range(n)])

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
        sat = False
        while not sat:
            n_rounds = 17
            n_bits = random.randint(self.opts.min_b, self.opts.max_b)

            bitsstr = '0b'+self.random_binary_string(512)
            
            cnf_filepath = os.path.abspath(os.path.join(sat_out_dir, '%.5d.cnf' % (i)))
            cmd_line = ['./cgen', 'encode', 'SHA1', '-vM', bitsstr, 'except:1..'+str(n_bits), \
                 '-vH', 'compute', '-r', str(n_rounds), cnf_filepath]
            
            try:
                process = subprocess.Popen(cmd_line, cwd=self.exec_dir, start_new_session=True)
                process.communicate()
            except:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            if not os.path.exists(cnf_filepath):
                continue
            
            n_vars, clauses = parse_cnf_file(cnf_filepath)
            clauses = clean_clauses(clauses)
            h = hash_clauses(clauses)

            if h in self.hash_list:
                continue

            solver = Cadical(bootstrap_with=clauses)
            sat = solver.solve()
            assert sat == True
            self.hash_list.append(h)
            write_dimacs_to(n_vars, clauses, cnf_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)

    parser.add_argument('--train_instances', type=int, default=0)
    parser.add_argument('--valid_instances', type=int, default=0)
    parser.add_argument('--test_instances', type=int, default=0)

    parser.add_argument('--min_b', type=int, default=5)
    parser.add_argument('--max_b', type=int, default=20)

    parser.add_argument('--seed', type=int, default=0)

    opts = parser.parse_args()

    random.seed(opts.seed)

    generator = Generator(opts)
    generator.run()


if __name__ == '__main__':
    main()
