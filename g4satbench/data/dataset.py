import os
import glob
import torch
import pickle
import itertools

from torch_geometric.data import Dataset
from g4satbench.utils.utils import parse_cnf_file, clean_clauses
from g4satbench.data.data import construct_lcg, construct_vcg


class SATDataset(Dataset):
    def __init__(self, data_dir, splits, sample_size, use_contrastive_learning, opts):
        self.opts = opts
        self.splits = splits
        self.sample_size = sample_size
        self.all_files = self._get_files(data_dir)
        self.split_len = self._get_split_len()
        self.all_labels = self._get_labels(data_dir)
        self.use_contrastive_learning = use_contrastive_learning
        if self.use_contrastive_learning:
            self.positive_indices = self._get_positive_indices()
            
        super().__init__(data_dir)
    
    def _get_files(self, data_dir):
        files = {}
        for split in self.splits:
            split_files = list(sorted(glob.glob(data_dir + f'/{split}/*.cnf', recursive=True)))
            if self.sample_size is not None and len(split_files) > self.sample_size:
                split_files = split_files[:self.sample_size]
            files[split] = split_files
        return files
    
    def _get_labels(self, data_dir):
        labels = {}
        if self.opts.label == 'satisfiability':
            for split in self.splits:
                if split == 'sat' or split == 'augmented_sat':
                    labels[split] = [torch.tensor(1., dtype=torch.float)] * self.split_len
                else:
                    # split == 'unsat' or split == 'augmented_unsat'
                    labels[split] = [torch.tensor(0., dtype=torch.float)] * self.split_len
        elif self.opts.label == 'assignment':
            for split in self.splits:
                assert split == 'sat' or split == 'augmented_sat'
                labels[split] = []
                for cnf_filepath in self.all_files[split]:
                    filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
                    assignment_file = os.path.join(os.path.dirname(cnf_filepath), filename + '_assignment.pkl')
                    with open(assignment_file, 'rb') as f:
                        assignment = pickle.load(f)
                    labels[split].append(torch.tensor(assignment, dtype=torch.float))
        elif self.opts.label == 'core_variable':
            for split in self.splits:
                assert split == 'unsat' or split == 'augmented_unsat'
                labels[split] = []
                for cnf_filepath in self.all_files[split]:
                    filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
                    assignment_file = os.path.join(os.path.dirname(cnf_filepath), filename + '_core_variable.pkl')
                    with open(assignment_file, 'rb') as f:
                        core_variable = pickle.load(f)
                    labels[split].append(torch.tensor(core_variable, dtype=torch.float))
        else:
            assert self.opts.label == None
            for split in self.splits:
                labels[split] = [None] * self.split_len
        
        return labels

    def _get_split_len(self):
        lens = [len(self.all_files[split]) for split in self.splits]
        assert len(set(lens)) == 1
        return lens[0]
    
    def _get_file_name(self, split, cnf_filepath):
        filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
        return f'{split}/{filename}_{self.opts.graph}.pt'
    
    def _get_positive_indices(self):
        # calculate the index to map the original instance to its augmented one, and vice versa.
        positive_indices = []
        for offset, split in enumerate(self.splits):
            if split == 'sat':
                positive_indices.append(torch.tensor(self.splits.index('augmented_sat')-offset, dtype=torch.long))
            elif split == 'augmented_sat':
                positive_indices.append(torch.tensor(self.splits.index('sat')-offset, dtype=torch.long))
            elif split == 'unsat':
                positive_indices.append(torch.tensor(self.splits.index('augmented_unsat')-offset, dtype=torch.long))
            elif split == 'augmented_unsat':
                positive_indices.append(torch.tensor(self.splits.index('unsat')-offset, dtype=torch.long))
        return positive_indices
    
    @property
    def processed_file_names(self):       
        names = []
        for split in self.splits:
            for cnf_filepath in self.all_files[split]:
                names.append(self._get_file_name(split, cnf_filepath))
        return names

    def _save_data(self, split, cnf_filepath):
        file_name = self._get_file_name(split, cnf_filepath)
        saved_path = os.path.join(self.processed_dir, file_name)
        if os.path.exists(saved_path):
            return
        
        n_vars, clauses, learned_clauses = parse_cnf_file(cnf_filepath, split_clauses=True)
        
        # limit the size of the learned clauses to 1000
        if len(learned_clauses) > 1000:
            clauses = clauses + learned_clauses[:1000]
        else:
            clauses = clauses + learned_clauses
        
        clauses = clean_clauses(clauses)
                    
        if self.opts.graph == 'lcg':
            data = construct_lcg(n_vars, clauses)
        elif self.opts.graph == 'vcg':
            data = construct_vcg(n_vars, clauses)

        torch.save(data, saved_path)
    
    def process(self):
        for split in self.splits:
            os.makedirs(os.path.join(self.processed_dir, split), exist_ok=True)
        
        for split in self.splits:
            for cnf_filepath in self.all_files[split]:
                self._save_data(split, cnf_filepath)
    
    def len(self):
        if self.opts.data_fetching == 'parallel':
            return self.split_len
        else:
            # self.opts.data_fetching == 'sequential'
            return self.split_len * len(self.splits)

    def get(self, idx):
        if self.opts.data_fetching == 'parallel':
            data_list = []
            for split_idx, split in enumerate(self.splits):
                cnf_filepath = self.all_files[split][idx]
                label = self.all_labels[split][idx]
                file_name = self._get_file_name(split, cnf_filepath)
                saved_path = os.path.join(self.processed_dir, file_name)
                data = torch.load(saved_path)
                data.y = label
                if self.use_contrastive_learning:
                    data.positive_index = self.positive_indices[split_idx]
                data_list.append(data)
            return data_list
        else:
            # self.opts.data_fetching == 'sequential'
            for split in self.splits:
                if idx >= self.split_len:
                    idx -= self.split_len
                else:
                    cnf_filepath = self.all_files[split][idx]
                    label = self.all_labels[split][idx]
                    file_name = self._get_file_name(split, cnf_filepath)
                    saved_path = os.path.join(self.processed_dir, file_name)
                    data = torch.load(saved_path)
                    data.y = label
                    return [data]
