import os
import argparse
import random
import math
import networkx as nx

from pysat.solvers import Cadical
from cnfgen import DominatingSet
from g4satbench.utils.utils import write_dimacs_to, VIG, clean_clauses, hash_clauses
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
                print(f'Generating k-domset {split} set...')
                for i in tqdm(range(n_instances)):
                    self.generate(i, sat_out_dir, unsat_out_dir)

    def generate(self, i, sat_out_dir, unsat_out_dir):
        sat = False
        unsat = False
        
        # ensure k is uniformly sampled
        k = random.randint(self.opts.min_k, self.opts.max_k)
        
        while not sat or not unsat:
            # randomly choose k
            v = random.randint(max(self.opts.min_v, k + 1), self.opts.max_v)
            # set p
            p = 1 - pow(1 - pow(1/math.comb(v,k), 1/(v-k)), 1/k)

            # randomly generate a graph
            graph = nx.generators.erdos_renyi_graph(v, p=p)
            if not nx.is_connected(graph):
                continue

            cnf = DominatingSet(graph, k)
            n_vars = len(list(cnf.variables()))
            clauses = list(cnf.clauses())
            clauses = [list(cnf._compress_clause(clause)) for clause in clauses]

            # ensure the graph in connected
            vig = VIG(n_vars, clauses)
            if not nx.is_connected(vig):
                continue
            
            # remove duplicate instances
            clauses = clean_clauses(clauses)
            h = hash_clauses(clauses)
            if h in self.hash_list:
                continue

            solver = Cadical(bootstrap_with=clauses)
            
            if solver.solve():
                if not sat:
                    sat = True
                    self.hash_list.append(h)
                    write_dimacs_to(n_vars, clauses, os.path.join(sat_out_dir, '%.5d.cnf' % (i)))
            else:
                if not unsat:
                    unsat = True
                    self.hash_list.append(h)
                    write_dimacs_to(n_vars, clauses, os.path.join(unsat_out_dir, '%.5d.cnf' % (i)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)

    parser.add_argument('--train_instances', type=int, default=0, help='The number of training instances')
    parser.add_argument('--valid_instances', type=int, default=0, help='The number of validating instances')
    parser.add_argument('--test_instances', type=int, default=0, help='The number of testing instances')
    
    parser.add_argument('--min_k', type=int, default=3, help='The minimum number for k')
    parser.add_argument('--max_k', type=int, default=5, help='The maximum number for k')

    parser.add_argument('--min_v', type=int, default=5, help='The minimum number of nodes in a instance')
    parser.add_argument('--max_v', type=int, default=20, help='The maximum number of nodes in a instance')

    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    opts = parser.parse_args()

    random.seed(opts.seed)

    generator = Generator(opts)
    generator.run()


if __name__ == '__main__':
    main()
