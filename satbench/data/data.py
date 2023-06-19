import torch

from torch_geometric.data import Data
from satbench.utils.utils import literal2l_idx, literal2v_idx


class LCG(Data):
    def __init__(self,
            l_size=None,
            c_size=None,
            l_edge_index=None,
            c_edge_index=None,
            l_batch=None,
            c_batch=None
        ):
        super().__init__()
        self.l_size = l_size
        self.c_size = c_size
        self.l_edge_index = l_edge_index
        self.c_edge_index = c_edge_index
        self.l_batch = l_batch
        self.c_batch = c_batch
       
    @property
    def num_edges(self):
        return self.c_edge_index.size(0)
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'l_edge_index':
            return self.l_size
        elif key == 'c_edge_index':
            return self.c_size
        elif key == 'l_batch' or key == 'c_batch' or key == 'positive_index':
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)


class VCG(Data):
    def __init__(self, 
            v_size=None,
            c_size=None,
            v_edge_index=None,
            c_edge_index=None,
            p_edge_index=None, 
            n_edge_index=None, 
            l_edge_index=None,
            v_batch=None,
            c_batch=None
        ):
        super().__init__()
        self.v_size = v_size
        self.c_size = c_size
        self.v_edge_index = v_edge_index
        self.c_edge_index = c_edge_index
        self.p_edge_index = p_edge_index
        self.n_edge_index = n_edge_index
        self.l_edge_index = l_edge_index
        self.v_batch = v_batch
        self.c_batch = c_batch
       
    @property
    def num_edges(self):
        return self.v_edge_index.size(0)
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'v_edge_index':
            return self.v_size
        elif key == 'c_edge_index':
            return self.c_size
        elif key == 'p_edge_index' or key == 'n_edge_index':
            return self.v_edge_index.size(0)
        elif key == 'l_edge_index':
            return self.v_size * 2
        elif key == 'v_batch' or key == 'c_batch' or key == 'positive_index':
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)


def construct_lcg(n_vars, clauses):
    l_edge_index_list = []
    c_edge_index_list = []
    
    for c_idx, clause in enumerate(clauses):
        for literal in clause:
            l_idx = literal2l_idx(literal)
            l_edge_index_list.append(l_idx)
            c_edge_index_list.append(c_idx)
    
    return LCG(
        n_vars * 2,
        len(clauses),
        torch.tensor(l_edge_index_list, dtype=torch.long),
        torch.tensor(c_edge_index_list, dtype=torch.long),
        torch.zeros(n_vars * 2, dtype=torch.long),
        torch.zeros(len(clauses), dtype=torch.long)
    )


def construct_vcg(n_vars, clauses):
    c_edge_index_list = []
    v_edge_index_list = []
    p_edge_index_list = []
    n_edge_index_list = []
    l_edge_index_list = []

    edge_index = 0
    for c_idx, clause in enumerate(clauses):
        for literal in clause:
            sign, v_idx = literal2v_idx(literal)
            c_edge_index_list.append(c_idx)
            v_edge_index_list.append(v_idx)
            
            if sign:
                p_edge_index_list.append(edge_index)
                l_edge_index_list.append(v_idx * 2)
            else:
                n_edge_index_list.append(edge_index)
                l_edge_index_list.append(v_idx * 2 + 1)
            
            edge_index += 1
    
    return VCG(
        n_vars,
        len(clauses),
        torch.tensor(v_edge_index_list, dtype=torch.long),
        torch.tensor(c_edge_index_list, dtype=torch.long),
        torch.tensor(p_edge_index_list, dtype=torch.long),
        torch.tensor(n_edge_index_list, dtype=torch.long),
        torch.tensor(l_edge_index_list, dtype=torch.long),
        torch.zeros(n_vars, dtype=torch.long),
        torch.zeros(len(clauses), dtype=torch.long)
    )
