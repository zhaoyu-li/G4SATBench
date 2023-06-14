import os
import argparse
import sympy
import networkx as nx

from pysat.solvers import Cadical
from satbench.utils.utils import ROOT_DIR, parse_cnf_file, VIG, clean_clauses, hash_clauses, write_dimacs_to 
from satbench.external.satfactor import generate_instance_known_factors
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
                print(f'Generating factoring {split} set...')
                for i in tqdm(range(n_instances)):
                    self.generate(i, sat_out_dir)

    def generate(self, i, sat_out_dir):
        sat = False
        
        while not sat:
            factor1 = sympy.randprime(pow(2, self.opts.min_b), pow(2, self.opts.max_b))
            factor2 = sympy.randprime(pow(2, self.opts.min_b), pow(2, self.opts.max_b))

            cnf_filepath = os.path.abspath(os.path.join(sat_out_dir, '%.5d.cnf' % (i)))
            dimacs = generate_instance_known_factors(factor1, factor2)
        
            with open(cnf_filepath, 'w') as f:
                f.write(dimacs)
        
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

    opts = parser.parse_args()

    generator = Generator(opts)
    generator.run()


if __name__ == '__main__':
    main()
