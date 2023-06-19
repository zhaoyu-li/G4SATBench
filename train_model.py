import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import argparse

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from satbench.utils.options import add_model_options
from satbench.utils.utils import set_seed, safe_log, safe_div
from satbench.utils.logger import Logger
from satbench.utils.format_print import FormatTable
from satbench.data.dataloader import get_dataloader
from satbench.models.gnn import GNN
from torch_scatter import scatter_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['satisfiability', 'assignment', 'core_variable'], help='Experiment task')
    parser.add_argument('train_dir', type=str, help='Directory with training data')
    parser.add_argument('--train_splits', type=str, nargs='+', choices=['sat', 'unsat', 'augmented_sat', 'augmented_unsat'], default=None, help='Category of the training data')
    parser.add_argument('--train_sample_size', type=int, default=None, help='The number of instance in training dataset')
    parser.add_argument('--checkpoint', type=str, default=None, help='pretrained checkpoint')
    parser.add_argument('--valid_dir', type=str, default=None, help='Directory with validating data')
    parser.add_argument('--valid_splits', type=str, nargs='+', choices=['sat', 'unsat', 'augmented_sat', 'augmented_unsat'], default=None, help='Category of the validating data')
    parser.add_argument('--valid_sample_size', type=int, default=None, help='The number of instance in validation dataset')
    parser.add_argument('--label', type=str, choices=[None, 'satisfiability', 'assignment', 'core_variable'], default=None, help='Label')
    parser.add_argument('--data_fetching', type=str, choices=['parallel', 'sequential'], default='parallel', help='Fetch data in sequential order or in parallel')
    parser.add_argument('--loss', type=str, choices=[None, 'supervised', 'unsupervised_1', 'unsupervised_2'], default=None, help='Loss type for assignment prediction')
    parser.add_argument('--save_model_epochs', type=int, default=1, help='Number of epochs between model savings')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs during training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='L2 regularization weight')
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler')
    parser.add_argument('--lr_step_size', type=int, default=50, help='Learning rate step size')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Learning rate factor')
    parser.add_argument('--lr_patience', type=int, default=10, help='Learning rate patience')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Clipping norm')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    add_model_options(parser)

    opts = parser.parse_args()

    set_seed(opts.seed)

    difficulty, dataset = tuple(os.path.abspath(opts.train_dir).split(os.path.sep)[-3:-1])
    splits_name = '_'.join(opts.train_splits)

    if opts.task == 'assignment':
        exp_name = f'train_task={opts.task}_difficulty={difficulty}_dataset={dataset}_splits={splits_name}_label={opts.label}_loss={opts.loss}/' + \
            f'graph={opts.graph}_init_emb={opts.init_emb}_model={opts.model}_n_iterations={opts.n_iterations}_lr={opts.lr}_weight_decay={opts.weight_decay}_seed={opts.seed}'
    else:
        exp_name = f'train_task={opts.task}_difficulty={difficulty}_dataset={dataset}_splits={splits_name}/' + \
            f'graph={opts.graph}_init_emb={opts.init_emb}_model={opts.model}_n_iterations={opts.n_iterations}_lr={opts.lr}_weight_decay={opts.weight_decay}_seed={opts.seed}'

    opts.log_dir = os.path.join('runs', exp_name)
    opts.checkpoint_dir = os.path.join(opts.log_dir, 'checkpoints')

    os.makedirs(opts.log_dir, exist_ok=True)
    os.makedirs(opts.checkpoint_dir, exist_ok=True)

    opts.log = os.path.join(opts.log_dir, 'log.txt')
    sys.stdout = Logger(opts.log, sys.stdout)
    sys.stderr = Logger(opts.log, sys.stderr)

    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opts)

    model = GNN(opts)
    model.to(opts.device)

    if opts.checkpoint is not None:
        print('Loading model checkpoint from %s..' % opts.checkpoint)
        if opts.device.type == 'cpu':
            checkpoint = torch.load(opts.checkpoint, map_location='cpu')
        else:
            checkpoint = torch.load(opts.checkpoint)

        model.load_state_dict(checkpoint['state_dict'], strict=False)

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    train_loader = get_dataloader(opts.train_dir, opts.train_splits, opts.train_sample_size, opts, 'train')

    if opts.valid_dir is not None:
        valid_loader = get_dataloader(opts.valid_dir, opts.valid_splits, opts.valid_sample_size, opts, 'valid')
    else:
        valid_loader = None

    if opts.scheduler is not None:
        if opts.scheduler == 'ReduceLROnPlateau':
            assert opts.valid_dir is not None
            scheduler = ReduceLROnPlateau(optimizer, factor=opts.lr_factor, patience=opts.lr_patience)
        else:
            assert opts.scheduler == 'StepLR'
            scheduler = StepLR(optimizer, step_size=opts.lr_step_size, gamma=opts.lr_factor)

    if opts.task == 'satisfiability' or opts.task == 'core_variable':
        format_table = FormatTable()

    best_loss = float('inf')
    for epoch in range(opts.epochs):
        print('EPOCH #%d' % epoch)
        print('Training...')
        train_loss = 0
        train_cnt = 0
        train_tot = 0

        if opts.task == 'satisfiability' or opts.task == 'core_variable':
            format_table.reset()

        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(opts.device)
            batch_size = data.num_graphs

            if opts.task == 'satisfiability':
                pred = model(data)
                label = data.y
                loss = F.binary_cross_entropy(pred, label)
                format_table.update(pred, label)

            elif opts.task == 'assignment':
                c_size = data.c_size.sum().item()
                c_batch = data.c_batch
                l_edge_index = data.l_edge_index
                c_edge_index = data.c_edge_index
                
                v_pred = model(data)

                if opts.loss == 'supervised':
                    label = data.y
                    loss = F.binary_cross_entropy(v_pred, label)

                elif opts.loss == 'unsupervised_1':
                    l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
                    s_max_denom = (l_pred[l_edge_index] / 0.1).exp()
                    s_max_nom = l_pred[l_edge_index] * s_max_denom

                    c_nom = scatter_sum(s_max_nom, c_edge_index, dim=0, dim_size=c_size)
                    c_denom = scatter_sum(s_max_denom, c_edge_index, dim=0, dim_size=c_size)
                    c_pred = safe_div(c_nom, c_denom)

                    s_min_denom = (-c_pred / 0.1).exp()
                    s_min_nom = c_pred * s_min_denom
                    s_nom = scatter_sum(s_min_nom, c_batch, dim=0, dim_size=batch_size)
                    s_denom = scatter_sum(s_min_denom, c_batch, dim=0, dim_size=batch_size)

                    score = safe_div(s_nom, s_denom)
                    loss = (1 - score).mean()

                elif opts.loss == 'unsupervised_2':
                    l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
                    l_pred_aggr = scatter_sum(safe_log(1 - l_pred[l_edge_index]), c_edge_index, dim=0, dim_size=c_size)
                    c_loss = -safe_log(1 - l_pred_aggr.exp())
                    loss = scatter_sum(c_loss, c_batch, dim=0, dim_size=batch_size).mean()
                    
                v_assign = (v_pred > 0.5).float()
                l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
                c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
                sat_batch = (scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size) == data.c_size).float()

                train_cnt += sat_batch.sum().item()
            
            else:
                assert opts.task == 'core_variable'
                v_pred = model(data)
                v_cls = v_pred > 0.5
                label = data.y
                loss = F.binary_cross_entropy(v_pred, label)

                format_table.update(v_pred, label)

            train_loss += loss.item() * batch_size
            train_tot += batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_norm)
            optimizer.step()

        train_loss /= train_tot
        print('Training LR: %f, Training loss: %f' % (optimizer.param_groups[0]['lr'], train_loss))

        if opts.task == 'satisfiability' or opts.task == 'core_variable':
            format_table.print_stats()
        else:
            assert opts.task == 'assignment'
            train_acc = train_cnt / train_tot
            print('Training accuracy: %f' % train_acc)

        if epoch % opts.save_model_epochs == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict()},
                os.path.join(opts.checkpoint_dir, 'model_%d.pt' % epoch)
            )

        if opts.valid_dir is not None:
            print('Validating...')
            valid_loss = 0
            valid_cnt = 0
            valid_tot = 0

            if opts.task == 'satisfiability' or opts.task == 'core_variable':
                format_table.reset()

            model.eval()
            for data in valid_loader:
                data = data.to(opts.device)
                batch_size = data.num_graphs
                with torch.no_grad():
                    if opts.task == 'satisfiability':
                        pred = model(data)
                        label = data.y
                        loss = F.binary_cross_entropy(pred, label)
                        format_table.update(pred, label)
                    
                    elif opts.task == 'assignment':
                        c_size = data.c_size.sum().item()
                        c_batch = data.c_batch
                        l_edge_index = data.l_edge_index
                        c_edge_index = data.c_edge_index

                        v_pred = model(data)

                        if opts.loss == 'supervised':
                            label = data.y
                            loss = F.binary_cross_entropy(v_pred, label)
                        
                        elif opts.loss == 'unsupervised_1':
                            l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
                            s_max_denom = (l_pred[l_edge_index] / 0.1).exp()
                            s_max_nom = l_pred[l_edge_index] * s_max_denom

                            c_nom = scatter_sum(s_max_nom, c_edge_index, dim=0, dim_size=c_size)
                            c_denom = scatter_sum(s_max_denom, c_edge_index, dim=0, dim_size=c_size)
                            c_pred = safe_div(c_nom, c_denom)

                            s_min_denom = (-c_pred / 0.1).exp()
                            s_min_nom = c_pred * s_min_denom
                            s_nom = scatter_sum(s_min_nom, c_batch, dim=0, dim_size=batch_size)
                            s_denom = scatter_sum(s_min_denom, c_batch, dim=0, dim_size=batch_size)

                            score = safe_div(s_nom, s_denom)
                            loss = (1 - score).mean()

                        elif opts.loss == 'unsupervised_2':
                            l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
                            l_pred_aggr = scatter_sum(safe_log(1 - l_pred[l_edge_index]), c_edge_index, dim=0, dim_size=c_size)
                            c_loss = -safe_log(1 - l_pred_aggr.exp())
                            loss = scatter_sum(c_loss, c_batch, dim=0, dim_size=batch_size).mean()

                        v_assign = (v_pred > 0.5).float()
                        l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
                        c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
                        sat_batch = (scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size) == data.c_size).float()
                        valid_cnt += sat_batch.sum().item()
                    
                    else:
                        assert opts.task == 'core_variable'
                        v_pred = model(data)
                        v_cls = v_pred > 0.5
                        label = data.y
                        loss = F.binary_cross_entropy(v_pred, label)

                        format_table.update(v_pred, label)

                valid_loss += loss.item() * batch_size
                valid_tot += batch_size

            valid_loss /= valid_tot
            print('Validating loss: %f' % valid_loss)

            if opts.task == 'satisfiability' or opts.task == 'core_variable':
                format_table.print_stats()
            else:
                assert opts.task == 'assignment'
                valid_acc = valid_cnt / valid_tot
                print('Validating accuracy: %f' % valid_acc)

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict()},
                    os.path.join(opts.checkpoint_dir, 'model_best.pt')
                )

            if opts.scheduler is not None:
                if opts.scheduler == 'ReduceLROnPlateau':
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()
        else:
            if opts.scheduler is not None:
                scheduler.step()


if __name__ == '__main__':
    main()
