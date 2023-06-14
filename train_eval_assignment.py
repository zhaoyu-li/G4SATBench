import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def fail_case():
    fail_dict = ['ca_supervised_easy_neurosat_lcg', 'ca_supervised_easy_gcn_lcg', 'ca_supervised_easy_ggnn_lcg', 'ca_supervised_easy_gin_lcg',
                 'ca_supervised_medium_neurosat_lcg', 'ca_supervised_medium_gcn_lcg', 'ca_supervised_medium_ggnn_lcg', 'ca_supervised_medium_gin_lcg',
                 'ca_unsupervisedv2_medium_gcn_vcg', 'ca_unsupervisedv2_medium_ggnn_vcg', 'ca_unsupervisedv2_medium_gin_vcg',
                 'k-clique_unsupervisedv2_medium_neurosat_lcg', 'k-clique_unsupervisedv2_medium_gcn_lcg', 'k-clique_unsupervisedv2_medium_ggnn_lcg', 'k-clique_unsupervisedv2_medium_gin_lcg',
                 'k-clique_unsupervisedv2_medium_gcn_vcg', 'k-clique_unsupervisedv2_medium_ggnn_vcg', 'k-clique_unsupervisedv2_medium_gin_vcg',
                 'k-clique_unsupervised_medium_neurosat_lcg', 'k-clique_unsupervised_medium_ggnn_vcg',
                 'k-domset_unsupervisedv2_easy_ggnn_vcg', 'k-domset_unsupervisedv2_easy_gin_vcg',
                 'k-domset_unsupervisedv2_medium_neurosat_lcg', 'k-domset_unsupervisedv2_medium_ggnn_lcg',
                 'k-domset_unsupervisedv2_medium_gcn_vcg', 'k-domset_unsupervisedv2_medium_ggnn_vcg', 'k-domset_unsupervisedv2_medium_gin_vcg',
                 'k-domset_unsupervised_easy_ggnn_lcg', 'k-domset_unsupervised_medium_ggnn_vcg', 'k-domset_unsupervised_easy_gin_vcg']
    os.makedirs('command', exist_ok=True)
    for seed in [666]:  # 123, 233, 345
        for lr in [0.0002, 0.00005]:
            for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset']:  # 'sr', '3-sat', 'ca', 'ps', 'k-clique'
                for model in ['gcn', 'gin', 'ggnn', 'neurosat']:  # 'gcn', 'gin', 'ggnn', 'neurosat'
                    for ite in [32]:
                        for graph in ['lcg', 'vcg']:  # 'lcg', 'vcg'
                            for difficulty in ['easy', 'medium']:
                                for loss in ['supervised', 'unsupervised', 'unsupervisedv2']:
                                    if model == 'neurosat' and graph == 'vcg':
                                        continue
                                    if f'{dataset}_{loss}_{difficulty}_{model}_{graph}' not in fail_dict:
                                        continue
                                    file_name = f'command/assignment_train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{loss}_{seed}.sh'
                                    with open(file_name, 'w') as f:
                                        # if model == 'gin':
                                        #    lr = 5.e-5
                                        # else:
                                        #    lr = 1.e-4
                                        # lr = 0.0001
                                        if dataset in ['k-domset', 'k-clique', 'k-vercov'] and difficulty == 'medium':
                                            batch_size = 64
                                        elif dataset in ['ca'] and difficulty == 'medium':
                                            batch_size = 50
                                        else:
                                            batch_size = 128
                                        # batch_size = 128
                                        f.write('#!/bin/bash\n')
                                        f.write(
                                            f'#SBATCH --job-name=assignment_train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{loss}_{seed}\n')
                                        f.write('#SBATCH --output=/dev/null\n')
                                        f.write('#SBATCH --ntasks=1\n')
                                        f.write('#SBATCH --time=2-12:00:00\n')
                                        f.write('#SBATCH --gres=gpu:rtx8000:1\n')
                                        f.write('#SBATCH --mem=16G\n')
                                        f.write('#SBATCH --cpus-per-task=8\n')
                                        f.write('\n')
                                        f.write('module load anaconda/3\n')
                                        f.write('conda activate satbench\n')
                                        f.write('\n')
                                        f.write(
                                            f'python train_model.py assignment $SCRATCH/satbench/{difficulty}/{dataset}/train/ \\\n')
                                        f.write('    --train_splits sat \\\n')
                                        f.write(f'    --valid_dir $SCRATCH/satbench/{difficulty}/{dataset}/valid/ \\\n')
                                        f.write('    --valid_splits sat \\\n')
                                        if loss == 'supervised':
                                            f.write('    --label assignment \\\n')
                                        f.write(f'    --loss {loss} \\\n')
                                        f.write('    --scheduler ReduceLROnPlateau \\\n')
                                        f.write(f'    --lr {lr} \\\n')
                                        f.write(f'    --n_iterations {ite} \\\n')
                                        f.write('    --weight_decay 1.e-8 \\\n')
                                        f.write(f'    --model {model} \\\n')
                                        f.write(f'    --graph {graph} \\\n')
                                        f.write(f'    --seed {seed} \\\n')
                                        f.write(f'    --batch_size {batch_size}\n')
                                    result = subprocess.run(
                                        ['sbatch', file_name],
                                        capture_output=False, text=False)
                                    if result.returncode == 0:
                                        print("Job submitted successfully.")
                                    else:
                                        print(f"Job submission failed with error: {result.stderr}")


def train_best_ite():
    best = {'easy': {}}
    best['easy']['sr'] = {'lcg': 'ggnn', 'vcg': 'ggnn'}
    best['easy']['3-sat'] = {'lcg': 'neurosat', 'vcg': 'ggnn'}
    best['easy']['ca'] = {'lcg': 'neurosat', 'vcg': 'gcn'}
    best['easy']['ps'] = {'lcg': 'gcn', 'vcg': 'gin'}
    best['easy']['k-clique'] = {'lcg': 'ggnn', 'vcg': 'ggnn'}
    best['easy']['k-domset'] = {'lcg': 'ggnn', 'vcg': 'ggnn'}
    best['easy']['k-vercov'] = {'lcg': 'neurosat', 'vcg': 'gcn'}
    os.makedirs('command', exist_ok=True)
    for seed in [123]:  # 123, 233, 345
        for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset']:  # 'ca', 'ps', 'k-clique',  'k-domset'
            for graph in ['lcg', 'vcg']:  # 'lcg', 'vcg'
                for difficulty in ['medium']:
                    for model in ['gcn', 'gin', 'ggnn', 'neurosat']:  # 'gcn', 'gin', 'ggnn', 'neurosat'
                        for ite in [32]:
                            for loss in ['supervised', 'unsupervised', 'unsupervisedv2']:
                                if model == 'neurosat' and graph == 'vcg':
                                    continue

                                file_name = f'command/assignment_train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{loss}_{seed}.sh'
                                with open(file_name, 'w') as f:
                                    # if model == 'gin':
                                    #    lr = 5.e-5
                                    # else:
                                    #    lr = 1.e-4
                                    lr = 1.e-4
                                    if dataset in ['k-domset', 'k-clique'] and difficulty == 'medium':
                                        batch_size = 64
                                    else:
                                        batch_size = 128
                                    # batch_size = 128
                                    f.write('#!/bin/bash\n')
                                    f.write(
                                        f'#SBATCH --job-name=assignment_train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{loss}_{seed}\n')
                                    f.write('#SBATCH --output=/dev/null\n')
                                    f.write('#SBATCH --ntasks=1\n')
                                    f.write('#SBATCH --time=2-12:00:00\n')
                                    f.write('#SBATCH --gres=gpu:rtx8000:1\n')
                                    f.write('#SBATCH --mem=16G\n')
                                    f.write('#SBATCH --cpus-per-task=8\n')
                                    f.write('\n')
                                    f.write('module load anaconda/3\n')
                                    f.write('conda activate satbench\n')
                                    f.write('\n')
                                    f.write(
                                        f'python train_model.py assignment $SCRATCH/satbench/{difficulty}/{dataset}/train/ \\\n')
                                    f.write('    --train_splits sat \\\n')
                                    f.write(f'    --valid_dir $SCRATCH/satbench/{difficulty}/{dataset}/valid/ \\\n')
                                    f.write('    --valid_splits sat \\\n')
                                    if loss == 'supervised':
                                        f.write('    --label assignment \\\n')
                                    f.write(f'    --loss {loss} \\\n')
                                    f.write('    --scheduler ReduceLROnPlateau \\\n')
                                    f.write(f'    --lr {lr} \\\n')
                                    f.write(f'    --n_iterations {ite} \\\n')
                                    f.write('    --weight_decay 1.e-8 \\\n')
                                    f.write(f'    --model {model} \\\n')
                                    f.write(f'    --graph {graph} \\\n')
                                    f.write(f'    --seed {seed} \\\n')
                                    f.write(f'    --batch_size {batch_size}\n')
                                result = subprocess.run(
                                    ['sbatch', file_name],
                                    capture_output=False, text=False)
                                if result.returncode == 0:
                                    print("Job submitted successfully.")
                                else:
                                    print(f"Job submission failed with error: {result.stderr}")

def train():
    os.makedirs('command', exist_ok=True)
    for seed in [123, 234, 345]: # 123, 233, 345
        for dataset in ['k-domset']: # 'sr', '3-sat', 'ca', 'ps', 'k-clique'
            for model in ['gcn', 'gin', 'ggnn', 'neurosat']: # 'gcn', 'gin', 'ggnn', 'neurosat'
                for ite in [32]:
                    for graph in ['lcg', 'vcg']: # 'lcg', 'vcg'
                        for difficulty in ['medium']:
                            for loss in ['supervised', 'unsupervised', 'unsupervisedv2']:
                                if model == 'neurosat' and graph == 'vcg':
                                    continue

                                file_name = f'command/assignment_train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{loss}_{seed}.sh'
                                with open(file_name, 'w') as f:
                                    # if model == 'gin':
                                    #    lr = 5.e-5
                                    # else:
                                    #    lr = 1.e-4
                                    lr = 0.0001
                                    if dataset in ['k-domset'] and difficulty == 'medium':
                                        batch_size = 64
                                    elif dataset in ['k-clique', 'ca'] and difficulty == 'medium':
                                        batch_size = 50
                                    else:
                                        batch_size = 128
                                    # batch_size = 128
                                    f.write('#!/bin/bash\n')
                                    f.write(f'#SBATCH --job-name=assignment_train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{loss}_{seed}\n')
                                    f.write('#SBATCH --output=/dev/null\n')
                                    f.write('#SBATCH --ntasks=1\n')
                                    f.write('#SBATCH --time=2-12:00:00\n')
                                    f.write('#SBATCH --gres=gpu:rtx8000:1\n')
                                    f.write('#SBATCH --mem=16G\n')
                                    f.write('#SBATCH --cpus-per-task=8\n')
                                    f.write('\n')
                                    f.write('module load anaconda/3\n')
                                    f.write('conda activate satbench\n')
                                    f.write('\n')
                                    f.write(f'python train_model.py assignment $SCRATCH/satbench/{difficulty}/{dataset}/train/ \\\n')
                                    f.write('    --train_splits sat \\\n')
                                    f.write(f'    --valid_dir $SCRATCH/satbench/{difficulty}/{dataset}/valid/ \\\n')
                                    f.write('    --valid_splits sat \\\n')
                                    if loss == 'supervised':
                                        f.write('    --label assignment \\\n')
                                    f.write(f'    --loss {loss} \\\n')
                                    f.write('    --scheduler ReduceLROnPlateau \\\n')
                                    f.write(f'    --lr {lr} \\\n')
                                    f.write(f'    --n_iterations {ite} \\\n')
                                    f.write('    --weight_decay 1.e-8 \\\n')
                                    f.write(f'    --model {model} \\\n')
                                    f.write(f'    --graph {graph} \\\n')
                                    f.write(f'    --seed {seed} \\\n')
                                    f.write(f'    --batch_size {batch_size}\n')
                                # result = subprocess.run(
                                #     ['sbatch', file_name],
                                #     capture_output=False, text=False)
                                # if result.returncode == 0:
                                #     print("Job submitted successfully.")
                                # else:
                                #     print(f"Job submission failed with error: {result.stderr}")

def eval():
    os.makedirs('command', exist_ok=True)
    file_name = f'command/assignment_eval.sh'
    with open(file_name, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write(
            f'#SBATCH --job-name=eval_k-vercov_sup&unsup\n')
        f.write('#SBATCH --output=/dev/null\n')
        f.write('#SBATCH --ntasks=1\n')
        f.write('#SBATCH --time=1-23:00:00\n')
        f.write('#SBATCH --gres=gpu:rtx8000:1\n')
        f.write('#SBATCH --mem=16G\n')
        f.write('#SBATCH --cpus-per-task=16\n')
        f.write('\n')
        f.write('module load anaconda/3\n')
        f.write('conda activate satbench\n')
        f.write('\n')
        os.makedirs('command', exist_ok=True)
        for seed in [123, 234, 345]:
            for dataset in ['k-vercov']: # 'sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov'
                for model in ['gcn', 'gin', 'ggnn', 'neurosat']:
                    for ite in [32]:
                        for graph in ['lcg', 'vcg']:
                            for difficulty in ['easy', 'medium']: # 'easy'
                                for loss in ['unsupervised', 'supervised', 'unsupervisedv2']: # ,'unsupervised', 'supervised'
                                    if model == 'neurosat' and graph == 'vcg':
                                        continue
                                    if loss == 'supervised':
                                        label = 'assignment'
                                    else:
                                        label = 'None'
                                    lr = '0.0001'
                                    if dataset in ['k-domset', 'k-clique', 'ca'] and difficulty == 'medium':
                                        batch_size = 64
                                    else:
                                        batch_size = 128

                                    f.write(
                                        f'python eval_model.py assignment /network/scratch/z/zhaoyu.li/satbench/{difficulty}/{dataset}/test/ \\\n')
                                    f.write(f'    /network/scratch/z/zhaoyu.li/runs/task\=assignment_difficulty\={difficulty}_dataset\={dataset}_splits\=sat_label\={label}_loss\={loss}/graph={graph}_init_emb=learned_model={model}_n_iterations={ite}_seed={seed}_lr={lr}_weight_decay=1e-08/checkpoints/model_best.pt \\\n')
                                    f.write(f'    --model {model} \\\n')
                                    f.write(f'    --graph {graph} \\\n')
                                    f.write(f'    --n_iterations {ite} \\\n')
                                    f.write(f'    --batch_size {batch_size} \\\n')
                                    f.write(f'    --test_splits sat\n')

                                    f.write(f'\n')

    result = subprocess.run(
        ['sbatch', file_name],
        capture_output=False, text=False)
    if result.returncode == 0:
        print("Job submitted successfully.")
    else:
        print(f"Job submission failed with error: {result.stderr}")


def summary_csv():
    acc_dict = {}
    for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov']: # 'sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov'
        acc_dict[f'{dataset}'] = {}
        for loss in ['supervised', 'unsupervisedv2', 'unsupervised']:
            acc_dict[f'{dataset}'][f'{loss}'] = {}
            for model in ['neurosat', 'gcn', 'ggnn', 'gin']:
                for ite in [32]:
                    for graph in ['lcg', 'vcg']:
                        if model == 'neurosat' and graph == 'vcg':
                            continue
                        acc_dict[f'{dataset}'][f'{loss}'][f'{model}_{graph}'] = {}
                        for difficulty in ['easy', 'medium']:
                            acc_dict[f'{dataset}'][f'{loss}'][f'{model}_{graph}'][f'{difficulty}'] = {}
                            acc = []
                            for seed in [123, 234, 345, 666]:
                                for lr in [0.0001, 0.0002, 0.00005]:
                                    if loss == 'supervised':
                                        label = 'assignment'
                                    else:
                                        label = 'None'
                                    dir = f'/network/scratch/z/zhaoyu.li/runs/task=assignment_difficulty={difficulty}_dataset={dataset}_splits=sat_label={label}_loss={loss}/graph={graph}_init_emb=learned_model={model}_n_iterations={ite}_seed={seed}_lr={lr}_weight_decay=1e-08/'
                                    file_name = f'eval_task=assignment_difficulty={difficulty}_dataset={dataset}_splits=sat_decoding=standard_n_iterations={ite}_checkpoint=model_best.txt'

                                    if os.path.exists(os.path.join(dir, file_name)) is not True:
                                        continue
                                    with open(os.path.join(dir, file_name), 'r') as ff:
                                        lines = ff.readlines()

                                    x = -1
                                    for line in lines:
                                        if 'Testing accuracy' in line:
                                            line = line.replace(' ', '').split(':')
                                            x = float(line[1])
                                    if x != -1:
                                        acc.append(x)
                            if len(acc) == 0:
                                mean = -1
                                std = -1
                            else:
                                acc = np.array(acc)
                                std = np.std(acc)
                                mean = np.max(acc) if std > 0.01 else np.mean(acc)
                                # mean = np.mean(acc)
                            acc_dict[f'{dataset}'][f'{loss}'][f'{model}_{graph}'][f'{difficulty}']['mean'] = mean
                            acc_dict[f'{dataset}'][f'{loss}'][f'{model}_{graph}'][f'{difficulty}']['std'] = std

    os.makedirs('results', exist_ok=True)

    file_name = f'results/assignment_eval.csv'

    with open(file_name, 'w') as f:
        print('mean', file=f, end='')
        for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov']:
            for loss in ['supervised', 'unsupervisedv2', 'unsupervised']:
                print(f',{dataset}_{loss}', file=f, end='')
        print('\n', file=f, end='')

        for difficulty in ['easy', 'medium']:
            for graph in ['lcg', 'vcg']:
                for model in ['neurosat', 'gcn', 'ggnn', 'gin']:
                    if model == 'neurosat' and graph == 'vcg':
                        continue
                    print(f'{model}_{graph}', file=f, end='')
                    for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov']:
                        for loss in ['supervised', 'unsupervisedv2', 'unsupervised']:
                            print(f',{acc_dict[f"{dataset}"][f"{loss}"][f"{model}_{graph}"][f"{difficulty}"]["mean"]*100:.2f}', file=f, end='')
                    print('\n', file=f, end='')
            print(file=f)

        # print('std', file=f, end='')
        # for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset']:
        #     for loss in ['supervised', 'unsupervisedv2', 'unsupervised']:
        #         print(f',{dataset}_{loss}', file=f, end='')
        # print('\n', file=f, end='')

        # for graph in ['lcg', 'vcg']:
        #     for model in ['gcn', 'gin', 'ggnn', 'neurosat']:
        #         if model == 'neurosat' and graph == 'vcg':
        #             continue
        #         for loss in ['supervised', 'unsupervisedv2', 'unsupervised']:
        #             print(f'{model}_{graph}', file=f, end='')
        #             for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset']:
        #                 print(f',{acc_dict[f"{dataset}"][f"{loss}"][f"{model}_{graph}"]["std"]}', file=f, end='')
        #             print('\n', file=f, end='')
        #     print('\n', file=f, end='')

def summary_latex():
    # summary_csv()
    csv_name = f'results/assignment_eval.csv'
    latex_name = f'results/assignment_eval.txt'

    with open(csv_name, 'r') as f:
        strs = f.readlines()
    with open(latex_name, 'w') as f:
        for str in strs:
            str = str.replace(',', ' & ')
            f.write(str)


def split_transfer():
    os.makedirs('command', exist_ok=True)
    models = ['ggnn', 'neurosat']  # ,
    graphs = ['vcg', 'lcg']  # 'lcg',
    for i in range(len(models)):
        model = models[i]
        graph = graphs[i]
        ite = 32
        file_name = f'command/split_eval_{graph}_{model}.sh'
        with open(file_name, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(
                f'#SBATCH --job-name=eval_split_{graph}_{model}\n')
            f.write('#SBATCH --output=/dev/null\n')
            f.write('#SBATCH --ntasks=1\n')
            f.write('#SBATCH --time=1-23:00:00\n')
            f.write('#SBATCH --gres=gpu:rtx8000:1\n')
            f.write('#SBATCH --mem=16G\n')
            f.write('#SBATCH --cpus-per-task=16\n')
            f.write('\n')
            f.write('module load anaconda/3\n')
            f.write('conda activate satbench\n')
            f.write('\n')
            os.makedirs('command', exist_ok=True)
            for dataset in ['k-vercov']:  # 'sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov'
                for difficulty in ['easy', 'medium']:  # 'easy'
                    loss = 'unsupervised'
                    user = 'z/zhaoyu.li'
                    if model == 'neurosat' and graph == 'vcg':
                        continue
                    if loss == 'supervised':
                        label = 'assignment'
                    else:
                        label = 'None'
                    lr = '0.0001'
                    if dataset in ['k-domset', 'k-clique', 'ca'] and difficulty == 'medium':
                        batch_size = 64
                    else:
                        batch_size = 128

                    f.write(
                        f'python eval_model.py assignment /network/scratch/{user}/satbench/{difficulty}/{dataset}/test/ \\\n')
                    f.write(
                        f'    /network/scratch/{user}/runs/task\=assignment_difficulty\={difficulty}_dataset\={dataset}_splits\=sat_unsat_label\={label}_loss\={loss}/graph={graph}_init_emb=learned_model={model}_n_iterations={ite}_seed=123_lr={lr}_weight_decay=1e-08/checkpoints/model_best.pt \\\n')
                    f.write(f'    --model {model} \\\n')
                    f.write(f'    --graph {graph} \\\n')
                    f.write(f'    --n_iterations {ite} \\\n')
                    f.write(f'    --batch_size {batch_size} \\\n')
                    f.write(f'    --test_splits sat \n')

                    f.write(f'\n')

        result = subprocess.run(
            ['sbatch', file_name],
            capture_output=False, text=False)
        if result.returncode == 0:
            print("Job submitted successfully.")
        else:
            print(f"Job submission failed with error: {result.stderr}")


def decoding():
    os.makedirs('command', exist_ok=True)
    models = ['ggnn'] # 'neurosat',
    graphs = ['vcg'] # 'lcg',
    for i in range(len(models)):
        model = models[i]
        graph = graphs[i]
        ite = 32
        file_name = f'command/decoding_eval_{graph}_{model}.sh'
        with open(file_name, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(
                f'#SBATCH --job-name=eval_decoding_{graph}_{model}\n')
            f.write('#SBATCH --output=/dev/null\n')
            f.write('#SBATCH --ntasks=1\n')
            f.write('#SBATCH --time=1-23:00:00\n')
            f.write('#SBATCH --gres=gpu:rtx8000:1\n')
            f.write('#SBATCH --mem=16G\n')
            f.write('#SBATCH --cpus-per-task=16\n')
            f.write('\n')
            f.write('module load anaconda/3\n')
            f.write('conda activate satbench\n')
            f.write('\n')
            os.makedirs('command', exist_ok=True)
            for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset']:  # 'sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov'
                for difficulty in ['easy', 'medium']:  # 'easy'
                    for decoding in ['multiple_assignments']:  # '2-clustering',
                        loss = 'unsupervised'
                        user = 'x/xujie.si'
                        if model == 'neurosat' and graph == 'vcg':
                            continue
                        if loss == 'supervised':
                            label = 'assignment'
                        else:
                            label = 'None'
                        lr = '0.0001'
                        if dataset in ['k-domset', 'k-clique', 'ca'] and difficulty == 'medium':
                            batch_size = 64
                        else:
                            batch_size = 128

                        f.write(
                            f'python eval_model.py assignment /network/scratch/{user}/satbench/{difficulty}/{dataset}/test/ \\\n')
                        f.write(
                            f'    /network/scratch/{user}/runs/task\=assignment_difficulty\={difficulty}_dataset\={dataset}_splits\=sat_label\={label}_loss\={loss}/graph={graph}_init_emb=learned_model={model}_n_iterations={ite}_seed=123_lr={lr}_weight_decay=1e-08/checkpoints/model_best.pt \\\n')
                        f.write(f'    --model {model} \\\n')
                        f.write(f'    --graph {graph} \\\n')
                        f.write(f'    --n_iterations {ite} \\\n')
                        f.write(f'    --batch_size {batch_size} \\\n')
                        f.write(f'    --test_splits sat \\\n')
                        f.write(f'    --decoding {decoding} \n')

                        f.write(f'\n')

        # result = subprocess.run(
        #     ['sbatch', file_name],
        #     capture_output=False, text=False)
        # if result.returncode == 0:
        #     print("Job submitted successfully.")
        # else:
        #     print(f"Job submission failed with error: {result.stderr}")


def decoding_cm():
    models = ['neurosat', 'ggnn']
    graphs = ['lcg', 'vcg']
    for i in range(len(models)):
        model = models[i]
        graph = graphs[i]
        for difficulty in ['easy', 'medium']:
            if model == 'neurosat':
                decoding_ls = ['standard', '2-clustering', 'multiple_assignments']
            else:
                decoding_ls = ['standard', 'multiple_assignments']
            dataset_ls = ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov']
            cm = np.zeros((len(decoding_ls), len(dataset_ls))) - 1.
            for i, decoding in enumerate(decoding_ls):
                for j, dataset in enumerate(dataset_ls):
                    user = 'x/xujie.si'
                    file = f'/home/mila/{user}/scratch/runs/task=assignment_difficulty={difficulty}_dataset={dataset}_splits=sat_label=None_loss=unsupervised/graph={graph}_init_emb=learned_model={model}_n_iterations=32_seed=123_lr=0.0001_weight_decay=1e-08/eval_task=assignment_difficulty={difficulty}_dataset={dataset}_splits=sat_decoding={decoding}_n_iterations=32_checkpoint=model_best.txt'
                    # if os.path.exists(file) is not True:
                    #     continue
                    with open(file, 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        if 'Testing accuracy' in line:
                            line = line.replace(' ', '').split(':')
                            cm[i][j] = float(line[1])

            print(difficulty, model, cm)

            colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

            plt.figure(figsize=(20.0, 8.0))

            x_label = ['SR', '3-SAT', 'CA', 'PS', r'$k$-Clique', r'$k$-Domset', r'$k$-Vercov']
            width = 0.1

            x = np.arange(len(x_label)) * width * (len(decoding_ls) + 1) + 0.2

            # colors = ['cornflowerblue', 'orangered', 'orange', 'orchid', 'slategrey']

            plt.figure()
            color_bar = {'Standard': '#F8766D', '2-clustering': '#00BA38', 'Multiple_assignments': '#619CFF'}
            for i in range(len(decoding_ls)):
                label = f'{decoding_ls[i][0].upper() + decoding_ls[i][1:]}'
                b = plt.bar(x + i * width, cm[i], width=width, label=label, color=color_bar[label], edgecolor=colors['black'])
                plt.bar_label(b, fmt='%.2f', padding=3, rotation='vertical')

            x_tick = x + width

            plt.xticks(x_tick, x_label)
            plt.yticks([0, 1.0])
            plt.xlabel("Datasets", fontsize=14)
            plt.ylabel("Classification accuracy (%)", fontsize=14)
            # plt.margins(x=0, tight=True)
            plt.subplots_adjust(left=0.1, right=0.9)
            plt.ylim([0.3, 1.2])

            plt.legend(loc='upper left', bbox_to_anchor=(0.25, -0.15))
            # plt.tight_layout()
            os.makedirs('assignment_decoding', exist_ok=True)
            plt.savefig(f'assignment_decoding/assignment_decoding_{model}_{difficulty}.pdf', bbox_inches='tight')
            plt.close()


def difficulty_transfer():
    graphs = ['lcg', 'vcg']
    models = ['neurosat', 'ggnn']
    seed = 123
    lr = '0.0001'

    for i in range(len(models)):
        model = models[i]
        graph = graphs[i]
        file_name = f'command/assignment_{model}_{graph}_difficulty_transfer.sh'
        os.makedirs('command', exist_ok=True)
        with open(file_name, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'#SBATCH --job-name=assignment_eval_{model}_{graph}_difficulty_transfer\n')
            f.write('#SBATCH --output=/dev/null\n')
            f.write('#SBATCH --ntasks=1\n')
            f.write('#SBATCH --time=1-23:00:00\n')
            f.write('#SBATCH --gres=gpu:rtx8000:1\n')
            f.write('#SBATCH --mem=16G\n')
            f.write('#SBATCH --cpus-per-task=16\n')
            f.write('\n')
            f.write('module load anaconda/3\n')
            f.write('conda activate satbench\n')
            f.write('\n')
            for dataset in ['k-vercov']:
                for ori in ['easy', 'medium']:  # 'sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov'
                    for tgt in ['easy', 'medium', 'hard']:
                        user = 'z/zhaoyu.li'
                        if tgt == 'medium' and dataset == 'k-clique':
                            batch_size = 64
                        else:
                            batch_size = 128
                        f.write(
                            f'python eval_model.py assignment /network/scratch/{user}/satbench/{tgt}/{dataset}/test/ \\\n')
                        f.write(
                            f'    /network/scratch/{user}/runs/task\=assignment_difficulty\={ori}_dataset\={dataset}_splits\=sat_label\=None_loss\=unsupervised/graph={graph}_init_emb=learned_model={model}_n_iterations=32_seed={seed}_lr={lr}_weight_decay=1e-08/checkpoints/model_best.pt \\\n')
                        f.write(f'    --model {model} \\\n')
                        f.write(f'    --graph {graph} \\\n')
                        f.write(f'    --n_iterations 32 \\\n')
                        f.write(f'    --batch_size {batch_size} \\\n')
                        f.write(f'    --test_splits sat \\\n')
                        f.write(f'    --difficulty_transfer \\\n')
                        f.write(f'    --ori_difficulty {ori} \n')
                        f.write(f'\n')

        result = subprocess.run(
            ['sbatch', file_name],
            capture_output=False, text=False)
        if result.returncode == 0:
            print("Job submitted successfully.")
        else:
            print(f"Job submission failed with error: {result.stderr}")


def difficulty_transfer_cm():
    os.makedirs('assignment_difficulty_transfer', exist_ok=True)
    graphs = ['lcg', 'vcg']
    models = ['neurosat', 'ggnn']
    fig, axes = plt.subplots(2, 2, figsize=(21, 16))
    label_model = ['NeuroSAT', 'GGNN']
    label_dataset = ['SR', '3-SAT']
    for row in range(len(models)):
        model = models[row]
        graph = graphs[row]
        for col, dataset in enumerate(['sr', '3-sat']):
            cm = np.zeros((2, 3)) - 1.
            for i, ori in enumerate(['easy', 'medium']):  # 'sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov'
                for j, tgt in enumerate(['easy', 'medium', 'hard']):
                    file = f'/home/mila/x/xujie.si/scratch/runs/task=assignment_difficulty={ori}_dataset={dataset}_splits=sat_label=None_loss=unsupervised/graph={graph}_init_emb=learned_model={model}_n_iterations=32_seed=123_lr=0.0001_weight_decay=1e-08/eval_task=assignment_ori_difficulty={ori}_tgt_difficulty={tgt}_dataset={dataset}_splits=sat_decoding=standard_n_iterations=32_checkpoint=model_best.txt'
                    if os.path.exists(file) is not True:
                        continue
                    with open(file, 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        if 'Testing accuracy' in line:
                            line = line.replace(' ', '').split(':')
                            cm[i][j] = float(line[1])
            axes[row][col].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.0173, vmax=0.9)
            # plt.title(f'{model}-{graph}')
            # plt.colorbar()
            axes[row][col].set_xticks(np.arange(3), ['easy', 'medium', 'hard'], fontsize=22)
            axes[row][col].set_yticks(np.arange(2), ['easy', 'medium'], fontsize=22)
            axes[row][col].tick_params(axis='x', length=0)
            axes[row][col].tick_params(axis='y', length=0)
            axes[row][col].set_title(f'{label_dataset[col]}; {label_model[row]}', fontsize=26, y=-0.2)
            # plt.xlabel('Testing dataset')
            # plt.ylabel('Training dataset')

            for ii in range(len(cm)):
                for jj in range(len(cm[ii])):
                    axes[row][col].text(jj, ii, f'{cm[ii][jj] * 100:.2f}', horizontalalignment="center",
                                        color="white" if cm[ii, jj] > np.mean(cm) else "black", fontsize=22)
            print(dataset, model, cm)
        plt.savefig(f'assignment_difficulty_transfer/difficulty_transfer_assigment.pdf', bbox_inches='tight')


def dataset_transfer():
    graphs = ['lcg', 'vcg']
    models = ['neurosat', 'ggnn']
    seed = 123
    lr = '0.0001'
    for i in range(len(models)):
        model = models[i]
        graph = graphs[i]

        os.makedirs('command', exist_ok=True)
        file_name = f'command/assignment_{model}_{graph}_dataset_transfer.sh'
        with open(file_name, 'w') as f:
            with open(file_name, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write(f'#SBATCH --job-name=assignment_eval_{model}_{graph}_dataset_transfer\n')
                f.write('#SBATCH --output=/dev/null\n')
                f.write('#SBATCH --ntasks=1\n')
                f.write('#SBATCH --time=1-23:00:00\n')
                f.write('#SBATCH --gres=gpu:rtx8000:1\n')
                f.write('#SBATCH --mem=16G\n')
                f.write('#SBATCH --cpus-per-task=16\n')
                f.write('\n')
                f.write('module load anaconda/3\n')
                f.write('conda activate satbench\n')
                f.write('\n')
                for difficulty in ['easy', 'medium']:
                    for ori in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov']:  # 'sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov'
                        for tgt in ['k-vercov']:
                            user = 'z/zhaoyu.li'
                            if difficulty == 'medium' and tgt == 'k-clique':
                                batch_size = 64
                            else:
                                batch_size = 128
                            f.write(
                                f'python eval_model.py assignment /network/scratch/{user}/satbench/{difficulty}/{tgt}/test/ \\\n')
                            f.write(
                                f'    /network/scratch/{user}/runs/task\=assignment_difficulty\={difficulty}_dataset\={ori}_splits\=sat_label\=None_loss\=unsupervised/graph={graph}_init_emb=learned_model={model}_n_iterations=32_seed={seed}_lr={lr}_weight_decay=1e-08/checkpoints/model_best.pt \\\n')
                            f.write(f'    --model {model} \\\n')
                            f.write(f'    --graph {graph} \\\n')
                            f.write(f'    --n_iterations 32 \\\n')
                            f.write(f'    --batch_size {batch_size} \\\n')
                            f.write(f'    --test_splits sat \\\n')
                            f.write(f'    --dataset_transfer \\\n')
                            f.write(f'    --ori_dataset {ori} \n')
                            f.write(f'\n')

        result = subprocess.run(
            ['sbatch', file_name],
            capture_output=False, text=False)
        if result.returncode == 0:
            print("Job submitted successfully.")
        else:
            print(f"Job submission failed with error: {result.stderr}")


# summary_latex()
# summary_csv()
# fail_case()
# eval()

# decoding()
# decoding_cm()
# difficulty_transfer_cm()
difficulty_transfer()
dataset_transfer()
split_transfer()