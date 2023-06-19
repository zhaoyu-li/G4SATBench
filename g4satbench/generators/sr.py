import os
import argparse
import numpy as np
import random
import networkx as nx

from pysat.solvers import Cadical
from ..utils.utils import write_dimacs_to, VIG, clean_clauses, hash_clauses
from tqdm import tqdm


class Generator:
    def __init__(self, opts):
        self.opts = opts
        self.hash_list = []

    def run(self):
        for split in ['train', 'valid', 'test']:
            n_instances = getattr(self.opts, f'{split}_instances')
            if n_instances > 0:
                sat_out_dir = os.path.join(os.path.abspath(self.opts.out_dir), f'{split}/sat')
                unsat_out_dir = os.path.join(os.path.abspath(self.opts.out_dir), f'{split}/unsat')
                os.makedirs(sat_out_dir, exist_ok=True)
                os.makedirs(unsat_out_dir, exist_ok=True)
                print(f'Generating sr {split} set...')
                for i in tqdm(range(n_instances)):
                    self.generate(i, sat_out_dir, unsat_out_dir)

    def generate(self, i, sat_out_dir, unsat_out_dir):
        while True:
            n_vars = random.randint(self.opts.min_n, self.opts.max_n)
            solver = Cadical()
            clauses = []
            while True:
                k_base = 1 if random.random() < self.opts.p_k_2 else 2
                k = k_base + np.random.geometric(self.opts.p_geo)
                vs = np.random.choice(n_vars, size=min(n_vars, k), replace=False)
                clause = [int(v + 1) if random.random() < 0.5 else int(-(v + 1)) for v in vs]

                solver.add_clause(clause)
                if solver.solve():
                    clauses.append(clause)
                else:
                    break
            
            unsat_clause = clause
            sat_clause = [-clause[0]] + clause[1:]

            clauses.append(unsat_clause)

            vig = VIG(n_vars, clauses)
            if not nx.is_connected(vig):
                continue

            clauses = clean_clauses(clauses)
            h = hash_clauses(clauses)

            if h not in self.hash_list:
                self.hash_list.append(h)
                break

        write_dimacs_to(n_vars, clauses, os.path.join(unsat_out_dir, '%.5d.cnf' % (i)))

        clauses[-1] = sat_clause
        write_dimacs_to(n_vars, clauses, os.path.join(sat_out_dir, '%.5d.cnf' % (i)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)
    
    parser.add_argument('--train_instances', type=int, default=0)
    parser.add_argument('--valid_instances', type=int, default=0)
    parser.add_argument('--test_instances', type=int, default=0)
    
    parser.add_argument('--min_n', type=int, default=10)
    parser.add_argument('--max_n', type=int, default=100)

    parser.add_argument('--p_k_2', type=float, default=0.3)
    parser.add_argument('--p_geo', type=float, default=0.4)

    parser.add_argument('--seed', type=int, default=0)

    opts = parser.parse_args()

    random.seed(opts.seed)

    generator = Generator(opts)
    generator.run()


if __name__ == '__main__':
    main()
