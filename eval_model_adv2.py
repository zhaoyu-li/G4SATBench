import torch
import torch.nn.functional as F
import os
import sys
import argparse
import pickle
import time

from satbench.utils.options import add_model_options
from satbench.utils.logger import Logger
from satbench.utils.utils import set_seed
from satbench.utils.format_print import FormatTable
from satbench.data.dataloader import get_dataloader
from satbench.models.gnn1 import GNN
from torch_scatter import scatter_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['satisfiability', 'assignment', 'core_variable'], help='Experiment task')
    parser.add_argument('test_dir', type=str, help='Directory with testing data')
    parser.add_argument('checkpoint', type=str, help='Checkpoint to be tested')
    parser.add_argument('--test_splits', type=str, nargs='+', choices=['sat', 'unsat', 'augmented_sat', 'augmented_unsat', 'trimmed'], default=None, help='Validation splits')
    parser.add_argument('--test_sample_size', type=int, default=None, help='The number of instance in validation dataset')
    parser.add_argument('--test_augment_ratio', type=float, default=None, help='The ratio between added clauses and all learned clauses')
    parser.add_argument('--label', type=str, choices=[None, 'satisfiability', 'core_variable'], default=None, help='Directory with validating data')
    parser.add_argument('--decoding', type=str, choices=['standard', '2-clustering', 'multiple_assignments'], default='standard', help='Decoding techniques for satisfying assignment prediction')
    parser.add_argument('--data_fetching', type=str, choices=['parallel', 'sequential'], default='parallel', help='Fetch data in sequential order or in parallel')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--dataset_transfer', action='store_true')
    parser.add_argument('--ori_dataset', type=str)
    parser.add_argument('--ite_transfer', action='store_true')
    parser.add_argument('--ori_ite', type=int)
    parser.add_argument('--difficulty_transfer', action='store_true')
    parser.add_argument('--ori_difficulty')

    add_model_options(parser)

    opts = parser.parse_args()

    set_seed(opts.seed)

    opts.log_dir = os.path.abspath(os.path.join(opts.checkpoint,  '..', '..'))

    difficulty, dataset = tuple(os.path.abspath(opts.test_dir).split(os.path.sep)[-3:-1])
    checkpoint_name = os.path.splitext(os.path.basename(opts.checkpoint))[0]
    names = []
    for split in opts.test_splits:
        if 'augment' in split and opts.test_augment_ratio is not None:
            names.append(split + str(opts.test_augment_ratio))
        else:
            names.append(split)
    splits_name = '_'.join(names)

    if opts.dataset_transfer:
        assert opts.ori_dataset is not None
        opts.log = os.path.join(opts.log_dir,
                                f'eval_task={opts.task}_difficulty={difficulty}_ori={opts.ori_dataset}_tgt={dataset}_splits={splits_name}_n_iterations={opts.n_iterations}_checkpoint={checkpoint_name}.txt')
    elif opts.ite_transfer:
        assert opts.ori_ite is not None
        opts.log = os.path.join(opts.log_dir, f'eval_task={opts.task}_difficulty={difficulty}_dataset={dataset}_splits={splits_name}_ori_ite={opts.ori_ite}_tgt_ite={opts.n_iterations}_checkpoint={checkpoint_name}.txt')
    elif opts.difficulty_transfer:
        assert opts.ori_difficulty is not None
        opts.log = os.path.join(opts.log_dir,
                                f'eval_task={opts.task}_ori_difficulty={opts.ori_difficulty}_tgt_difficulty={difficulty}_dataset={dataset}_splits={splits_name}_eval_ite_{opts.n_iterations}_checkpoint={checkpoint_name}.txt')
    else:
        if opts.task == 'assignment':
            opts.log = os.path.join(opts.log_dir, f'eval_task={opts.task}_difficulty={difficulty}_dataset={dataset}_splits={splits_name}_decoding={opts.decoding}_n_iterations={opts.n_iterations}_checkpoint={checkpoint_name}.txt')
        else:
            opts.log = os.path.join(opts.log_dir, f'eval_task={opts.task}_difficulty={difficulty}_dataset={dataset}_splits={splits_name}_n_iterations={opts.n_iterations}_checkpoint={checkpoint_name}.txt')

    sys.stdout = Logger(opts.log, sys.stdout)
    sys.stderr = Logger(opts.log, sys.stderr)

    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opts)

    model = GNN(opts)
    model.to(opts.device)
    test_loader = get_dataloader(opts.test_dir, opts.test_splits, opts.test_sample_size, opts.test_augment_ratio, opts, 'test')

    print('Loading model checkpoint from %s..' % opts.checkpoint)
    if opts.device.type == 'cpu':
        checkpoint = torch.load(opts.checkpoint, map_location='cpu')
    else:
        checkpoint = torch.load(opts.checkpoint)

    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to(opts.device)

    test_tot = 0
    test_cnt = 0

    if opts.task == 'satisfiability' or opts.task == 'core_variable':
        format_table = FormatTable()

    t0 = time.time()

    number_unsat_clauses = [0 for _ in range(opts.n_iterations + 1)]

    print('Testing...')
    model.eval()
    for data in test_loader:
        data = data.to(opts.device)
        batch_size = data.num_graphs
        with torch.no_grad():
            if opts.task == 'satisfiability':
                pred = model(data)
                label = data.y
                format_table.update(pred, label)
            elif opts.task == 'assignment':
                c_size = data.c_size.sum().item()
                c_batch = data.c_batch
                l_edge_index = data.l_edge_index
                c_edge_index = data.c_edge_index
                if opts.decoding == 'standard':
                    v_pred = model(data)
                    v_assign = (v_pred > 0.5).float()
                    l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
                    c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
                    sat_batch = (scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size) == data.c_size).float()
                    test_cnt += sat_batch.sum().item()
                elif opts.decoding == '2-clustering':
                    ass = model(data)
                    for iter, v_assigns in enumerate(ass):
                        unsat_clause_list = []
                        for v_assign in v_assigns:
                            l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
                            c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
                            unsat_clauses = data.c_size - scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size)
                            unsat_clause_list.append(unsat_clauses)

                        unsat_clause_tensor = torch.stack(unsat_clause_list, dim=1)
                        unsat_clauses = torch.min(unsat_clause_tensor, dim=1)[0]
                        number_unsat_clauses[iter] += unsat_clauses.sum().item()

                else:
                    assert opts.decoding == 'multiple_assignments'
                    v_preds = model(data)
                    sat_batches = []
                    assignments = [[] for _ in range(batch_size)]
                    for iter, v_pred in enumerate(v_preds):
                        v_assign = (v_pred > 0.5).float()
                        v_assign_list = v_assign.tolist()
                        l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
                        c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
                        unsat_clauses = data.c_size - scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size)
                        number_unsat_clauses[iter] += unsat_clauses.sum().item()
                        sat_batches.append((scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size) == data.c_size).float())
                    sat_batch = torch.clamp(torch.stack(sat_batches, dim=0).sum(dim=0), max=1)
                    test_cnt += sat_batch.sum().item()

                test_tot += batch_size
            else:
                assert opts.task == 'core_variable'
                v_pred = model(data)
                label = data.y
                format_table.update(v_pred, label)

    if opts.task == 'satisfiability' or opts.task == 'core_variable':
        format_table.print_stats()
    elif opts.task == 'assignment':
        test_acc = test_cnt / test_tot
        print('Testing accuracy: %f' % test_acc)

    number_unsat_clauses = torch.tensor(number_unsat_clauses) / test_tot
    print(number_unsat_clauses)

    t = time.time() - t0
    print('Solving Time: %f' % t)


if __name__ == '__main__':
    main()
