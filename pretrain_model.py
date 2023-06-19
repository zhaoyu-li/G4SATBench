import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import argparse

from satbench.utils.options import add_model_options
from satbench.utils.utils import set_seed, safe_log, safe_div
from satbench.utils.logger import Logger
from satbench.data.dataloader import get_dataloader
from satbench.models.gnn import GNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', type=str, help='Directory with training data')
    parser.add_argument('--task', type=str, choices=['satisfiability'], default='satisfiability', help='Experiment task')
    parser.add_argument('--train_splits', type=str, nargs='+', choices=['sat', 'unsat', 'augmented_sat', 'augmented_unsat'], default=None, help='Category of the training data')
    parser.add_argument('--train_sample_size', type=int, default=None, help='The number of instance in training dataset')
    parser.add_argument('--train_augment_ratio', type=float, default=None, help='The ratio between added clauses and all learned clauses')
    parser.add_argument('--use_contrastive_learning', type=bool, choices=[True], default=True)
    parser.add_argument('--label', type=str, choices=None, default=None, help='Label')
    parser.add_argument('--data_fetching', type=str, choices=['parallel', 'sequential'], default='parallel', help='Fetch data in sequential order or in parallel')
    parser.add_argument('--save_model_epochs', type=int, default=1, help='Number of epochs between model savings')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs during training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='L2 regularization weight')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Clipping norm')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    add_model_options(parser)

    opts = parser.parse_args()

    set_seed(opts.seed)

    difficulty, dataset = tuple(os.path.abspath(opts.train_dir).split(os.path.sep)[-3:-1])
    splits_name = '_'.join(train_splits)
    exp_name = f'pretrain_task={opts.task}_difficulty={difficulty}_dataset={dataset}_splits={splits_name}/' + \
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

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    train_loader = get_dataloader(opts.train_dir, opts.train_splits, opts.train_sample_size, opts, 'train', self.opts.use_contrastive_learning) # use contrastive learning

    best_loss = float('inf')
    for epoch in range(opts.epochs):
        print('EPOCH #%d' % epoch)
        print('Training...')
        train_loss = 0
        train_tot = 0

        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(opts.device)
            batch_size = data.num_graphs

            sim = model(data)
            positive_index = data.positive_index
            loss = -safe_log(sim[torch.arange(batch_size), data.positive_index] / sim.sum(dim=1)).mean()
            
            train_loss += loss.item() * batch_size
            train_tot += batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_norm)
            optimizer.step()

        train_loss /= train_tot
        print('Training LR: %f, Training loss: %f' % (optimizer.param_groups[0]['lr'], train_loss))

        if epoch % opts.save_model_epochs == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict()},
                os.path.join(opts.checkpoint_dir, 'model_%d.pt' % epoch)
            )
        
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict()},
                os.path.join(opts.checkpoint_dir, 'model_best.pt')
            )


if __name__ == '__main__':
    main()
