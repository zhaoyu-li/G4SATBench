import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import csv


def fail_case():
    models = ['neurosat', 'neurosat', 'ggnn', 'ggnn', 'ggnn']
    graphs = ['lcg', 'lcg', 'lcg', 'vcg', 'vcg']
    datasets = ['ca', 'k-clique', 'ca', 'ca', 'k-clique']
    for i in range(len(models)):
        model = models[i]
        graph = graphs[i]
        dataset = datasets[i]
        file_name = f'command/satisfiability_train_{model}_32_{graph}_{dataset}_medium_123.sh'
        with open(file_name, 'w') as f:
            lr = 0.0001
            batch_size = 32
            # batch_size = 128
            f.write('#!/bin/bash\n')
            f.write(f'#SBATCH --job-name=train_{model}_32_{graph}_{dataset}_medium_123\n')
            f.write('#SBATCH --output=/dev/null\n')
            f.write('#SBATCH --ntasks=1\n')
            f.write('#SBATCH --time=3-23:00:00\n')
            f.write('#SBATCH --gres=gpu:rtx8000:1\n')
            f.write('#SBATCH --mem=16G\n')
            f.write('#SBATCH --cpus-per-task=8\n')
            f.write('\n')
            f.write('module load anaconda/3\n')
            f.write('conda activate satbench\n')
            f.write('\n')
            f.write(f'python train_model.py satisfiability $SCRATCH/satbench/medium/{dataset}/train/ \\\n')
            f.write('    --train_splits sat unsat \\\n')
            f.write(f'    --valid_dir $SCRATCH/satbench/medium/{dataset}/valid/ \\\n')
            f.write('    --valid_splits sat unsat \\\n')
            f.write('    --label satisfiability \\\n')
            f.write('    --scheduler ReduceLROnPlateau \\\n')
            f.write(f'    --lr {lr} \\\n')
            f.write(f'    --n_iterations 32 \\\n')
            f.write('    --weight_decay 1.e-8 \\\n')
            f.write(f'    --model {model} \\\n')
            f.write(f'    --graph {graph} \\\n')
            f.write(f'    --epochs 100 \\\n')
            f.write(f'    --seed 123 \\\n')
            f.write(f'    --batch_size {batch_size}\n')
        result = subprocess.run(
            ['sbatch', file_name],
            capture_output=False, text=False)
        if result.returncode == 0:
            print("Job submitted successfully.")
        else:
            print(f"Job submission failed with error: {result.stderr}")



def train_best_ite():
    # best = {'easy': {}}
    # best['easy']['sr'] = {'lcg': 'ggnn', 'vcg': 'ggnn'}
    # best['easy']['3-sat'] = {'lcg': 'neurosat', 'vcg': 'ggnn'}
    # best['easy']['ca'] = {'lcg': 'neurosat', 'vcg': 'gcn'}
    # best['easy']['ps'] = {'lcg': 'gcn', 'vcg': 'gin'}
    # best['easy']['k-clique'] = {'lcg': 'ggnn', 'vcg': 'ggnn'}
    # best['easy']['k-domset'] = {'lcg': 'ggnn', 'vcg': 'ggnn'}
    # best['easy']['k-vercov'] = {'lcg': 'neurosat', 'vcg': 'gcn'}
    best = {'lcg': 'neurosat', 'vcg': 'ggnn'}
    os.makedirs('command', exist_ok=True)
    for seed in [123]: # 123, 233, 345
        for dataset in ['sr', '3-sat', 'k-clique']: # 'ca', 'ps', 'k-clique',  'k-domset', 'k-color'
            for graph in ['lcg', 'vcg']:  # 'lcg', 'vcg'
                for difficulty in ['easy', 'medium']:
                    for ite in [64]: # 8, 16
                            model = best[graph]
                            file_name = f'command/satisfiability_best_train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{seed}.sh'
                            with open(file_name, 'w') as f:
                                lr = 0.0001
                                if difficulty == 'medium' and dataset in ['sr', '3-sat']:
                                    batch_size = 32
                                else:
                                    batch_size = 25

                                f.write('#!/bin/bash\n')
                                f.write(f'#SBATCH --job-name=train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{seed}\n')
                                f.write('#SBATCH --output=/dev/null\n')
                                f.write('#SBATCH --ntasks=1\n')
                                f.write('#SBATCH --time=3-23:00:00\n')
                                f.write('#SBATCH --gres=gpu:rtx8000:1\n')
                                f.write('#SBATCH --mem=16G\n')
                                f.write('#SBATCH --cpus-per-task=8\n')
                                f.write('\n')
                                f.write('module load anaconda/3\n')
                                f.write('conda activate satbench\n')
                                f.write('\n')
                                f.write(f'python train_model.py satisfiability $SCRATCH/satbench/{difficulty}/{dataset}/train/ \\\n')
                                f.write('    --train_splits sat unsat \\\n')
                                f.write(f'    --valid_dir $SCRATCH/satbench/{difficulty}/{dataset}/valid/ \\\n')
                                f.write('    --valid_splits sat unsat \\\n')
                                f.write('    --label satisfiability \\\n')
                                f.write('    --scheduler ReduceLROnPlateau \\\n')
                                f.write(f'    --lr {lr} \\\n')
                                f.write(f'    --n_iterations {ite} \\\n')
                                f.write('    --weight_decay 1.e-8 \\\n')
                                f.write(f'    --model {model} \\\n')
                                f.write(f'    --graph {graph} \\\n')
                                f.write(f'    --seed {seed} \\\n')
                                f.write(f'    --epochs 50 \\\n')
                                f.write(f'    --batch_size {batch_size}\n')
                            result = subprocess.run(
                                ['sbatch', f'command/satisfiability_best_train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{seed}.sh'],
                                capture_output=False, text=False)
                            if result.returncode == 0:
                                print("Job submitted successfully.")
                            else:
                                print(f"Job submission failed with error: {result.stderr}")


def train():
    os.makedirs('command', exist_ok=True)
    for seed in [123, 234, 345]: # 123, 233, 345
        for dataset in ['k-clique', 'k-domset']: # 'ca', 'ps', 'k-clique',  'k-domset', 'k-color'
            for model in ['gcn', 'gin', 'ggnn', 'neurosat']: # 'gcn', 'gin', 'ggnn', 'neurosat'
                for ite in [32]:
                    for graph in ['lcg', 'vcg']: # 'lcg', 'vcg'
                        for difficulty in ['medium']:
                            if model == 'neurosat' and graph == 'vcg':
                                continue
                            with open(f'command/satisfiability_train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{seed}.sh', 'w') as f:
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
                                f.write(f'#SBATCH --job-name=train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{seed}\n')
                                f.write('#SBATCH --output=/dev/null\n')
                                f.write('#SBATCH --ntasks=1\n')
                                f.write('#SBATCH --time=3-23:00:00\n')
                                f.write('#SBATCH --gres=gpu:rtx8000:1\n')
                                f.write('#SBATCH --mem=16G\n')
                                f.write('#SBATCH --cpus-per-task=8\n')
                                f.write('\n')
                                f.write('module load anaconda/3\n')
                                f.write('conda activate satbench\n')
                                f.write('\n')
                                f.write(f'python train_model.py satisfiability $SCRATCH/satbench/{difficulty}/{dataset}/train/ \\\n')
                                f.write('    --train_splits sat unsat \\\n')
                                f.write(f'    --valid_dir $SCRATCH/satbench/{difficulty}/{dataset}/valid/ \\\n')
                                f.write('    --valid_splits sat unsat \\\n')
                                f.write('    --label satisfiability \\\n')
                                f.write('    --scheduler ReduceLROnPlateau \\\n')
                                f.write(f'    --lr {lr} \\\n')
                                f.write(f'    --n_iterations {ite} \\\n')
                                f.write('    --weight_decay 1.e-8 \\\n')
                                f.write(f'    --model {model} \\\n')
                                f.write(f'    --graph {graph} \\\n')
                                f.write(f'    --seed {seed} \\\n')
                                f.write(f'    --batch_size {batch_size}\n')
                            result = subprocess.run(
                                ['sbatch', f'command/satisfiability_train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{seed}.sh'],
                                capture_output=False, text=False)
                            if result.returncode == 0:
                                print("Job submitted successfully.")
                            else:
                                print(f"Job submission failed with error: {result.stderr}")


def eval():
    file_name = f'command/satisfiability_eval.sh'
    with open(file_name, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write(
            f'#SBATCH --job-name=eval\n')
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
            for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset']: # 'sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov'
                for model in ['gcn', 'gin', 'ggnn', 'neurosat']:
                    for ite in [32]:
                        for graph in ['lcg', 'vcg']:
                            for difficulty in ['medium']: # 'easy'
                                if model == 'neurosat' and graph == 'vcg':
                                    continue

                                lr = '0.0001'
                                if dataset in ['k-domset', 'k-clique'] and difficulty == 'medium':
                                    batch_size = 64
                                else:
                                    batch_size = 128

                                f.write(
                                    f'python eval_model.py satisfiability /network/scratch/x/xujie.si/satbench/{difficulty}/{dataset}/test/ \\\n')
                                f.write(f'    /network/scratch/x/xujie.si/runs/task\=satisfiability_difficulty\={difficulty}_dataset\={dataset}_splits\=sat_unsat_label\=satisfiability_loss\=None/graph={graph}_init_emb=learned_model={model}_n_iterations={ite}_seed={seed}_lr={lr}_weight_decay=1e-08/checkpoints/model_best.pt \\\n')
                                f.write(f'    --model {model} \\\n')
                                f.write(f'    --graph {graph} \\\n')
                                f.write(f'    --n_iterations {ite} \\\n')
                                f.write(f'    --batch_size {batch_size} \\\n')
                                f.write(f'    --label satisfiability \\\n')
                                f.write(f'    --test_splits sat unsat\n')
                                f.write(f'\n')

    result = subprocess.run(
        ['sbatch', f'command/satisfiability_eval.sh'],
        capture_output=False, text=False)
    if result.returncode == 0:
        print("Job submitted successfully.")
    else:
        print(f"Job submission failed with error: {result.stderr}")


def summary_csv():
    acc_dict = {}
    for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov']: # 'sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov'
        acc_dict[f'{dataset}'] = {}
        for model in ['gcn', 'gin', 'ggnn', 'neurosat']:
            for ite in [32]:
                for graph in ['lcg', 'vcg']:
                    if model == 'neurosat' and graph == 'vcg':
                        continue
                    acc_dict[f'{dataset}'][f'{model}_{graph}'] = {}
                    for difficulty in ['easy', 'medium']:
                        acc_dict[f'{dataset}'][f'{model}_{graph}'][difficulty] = {}
                        acc = []
                        for seed in [123, 234, 345]:
                            lr = '0.0001'
                            dir = f'/network/scratch/x/xujie.si/runs/task=satisfiability_difficulty={difficulty}_dataset={dataset}_splits=sat_unsat_label=satisfiability_loss=None/graph={graph}_init_emb=learned_model={model}_n_iterations={ite}_seed={seed}_lr={lr}_weight_decay=1e-08/'
                            file_name = f'eval_task=satisfiability_difficulty={difficulty}_dataset={dataset}_splits=sat_unsat_n_iterations={ite}_checkpoint=model_best.txt'

                            if os.path.exists(os.path.join(dir, file_name)) is not True:
                                continue
                            with open(os.path.join(dir, file_name), 'r') as ff:
                                lines = ff.readlines()

                            x = -1
                            for line in lines:
                                if 'Overall accuracy' in line:
                                    line = line.replace(' ', '').split('|')
                                    x = float(line[2])
                            if x != -1:
                                acc.append(x)

                        if len(acc) == 0:
                            mean = -1
                            std = -1
                        else:
                            acc = np.array(acc)
                            mean = np.mean(acc)
                            std = np.std(acc)

                        acc_dict[f'{dataset}'][f'{model}_{graph}'][difficulty]['mean'] = mean
                        acc_dict[f'{dataset}'][f'{model}_{graph}'][difficulty]['std'] = std

    os.makedirs('results', exist_ok=True)

    file_name = f'results/satisfiability_eval.csv'

    with open(file_name, 'w') as f:
        print('mean', file=f, end='')
        for difficulty in ['easy', 'medium']:
            for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov']:
                print(f',{dataset}_{difficulty}', file=f, end='')
            print('\n', file=f, end='')

        for graph in ['lcg', 'vcg']:
            for model in ['neurosat', 'gcn', 'ggnn', 'gin']:
                if model == 'neurosat' and graph == 'vcg':
                    continue
                print(f'{model}_{graph}', file=f, end='')
                for difficulty in ['easy', 'medium']:
                    for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov']:
                        print(f',{acc_dict[f"{dataset}"][f"{model}_{graph}"][difficulty]["mean"]*100:.2f}', file=f, end='')
                print('\n', file=f, end='')

        # print(file=f)
        # print('std', file=f, end='')
        # for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset']:
        #     print(f',{dataset}', file=f, end='')
        # print('\n', file=f, end='')
        #
        # for graph in ['lcg', 'vcg']:
        #     for model in ['neurosat', 'gcn', 'ggnn', 'gin']:
        #         if model == 'neurosat' and graph == 'vcg':
        #             continue
        #         print(f'{model}_{graph}', file=f, end='')
        #         for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset']:
        #             print(f',{acc_dict[f"{dataset}"][f"{model}_{graph}"]["std"]}', file=f, end='')
        #         print('\n', file=f, end='')

def summary_latex():
    summary_csv()
    csv_name = f'results/satisfiability_eval.csv'
    latex_name = f'results/satisfiability_eval.txt'
    if os.path.exists(latex_name):
        os.system(f'rm {latex_name}')
    with open(csv_name, 'r') as f:
        for str in f.readlines():
            with open(latex_name, 'a') as ff:
                str = str.replace(',', ' & ')
                ff.write(str)

def dataset_transfer():
    graphs = ['lcg', 'vcg']
    models = ['neurosat', 'ggnn']
    ite = 32
    seed = 123
    lr = '0.0001'
    for i in range(len(models)):
        model = models[i]
        graph = graphs[i]

        os.makedirs('command', exist_ok=True)
        file_name = f'command/satisfiability_{model}_{graph}_dataset_transfer.sh'
        with open(file_name, 'w') as f:
            f.write('#!/bin/bash\n')
            # f.write(f'#SBATCH --job-name=eval_dataset_transfer\n')
            # f.write('#SBATCH --output=/dev/null\n')
            # f.write('#SBATCH --ntasks=1\n')
            # f.write('#SBATCH --time=1-23:00:00\n')
            # f.write('#SBATCH --gres=gpu:rtx8000:1\n')
            # f.write('#SBATCH --mem=16G\n')
            # f.write('#SBATCH --cpus-per-task=16\n')
            # f.write('\n')
            # f.write('module load anaconda/3\n')
            # f.write('conda activate satbench\n')
            # f.write('\n')
            for ori in ['k-vercov']:  # 'sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov'
                for tgt in ['k-vercov']:
                    for difficulty in ['easy', 'medium']:
                        if difficulty == 'medium' and tgt in ['ca', 'k-clique']:
                            batch_size =50
                        else:
                            batch_size = 128
                        user = 'z/zhaoyu.li'
                        f.write(f'python eval_model.py satisfiability /network/scratch/{user}/satbench/{difficulty}/{tgt}/test/ \\\n')
                        f.write(f'    /network/scratch/{user}/runs/task\=satisfiability_difficulty\={difficulty}_dataset\={ori}_splits\=sat_unsat_label\=satisfiability_loss\=None/graph={graph}_init_emb=learned_model={model}_n_iterations={ite}_seed={seed}_lr={lr}_weight_decay=1e-08/checkpoints/model_best.pt \\\n')
                        f.write(f'    --model {model} \\\n')
                        f.write(f'    --graph {graph} \\\n')
                        f.write(f'    --n_iterations {ite} \\\n')
                        f.write(f'    --batch_size {batch_size} \\\n')
                        f.write(f'    --label satisfiability \\\n')
                        f.write(f'    --test_splits sat unsat \\\n')
                        f.write(f'    --dataset_transfer \\\n')
                        f.write(f'    --ori_dataset {ori} \n')
                        f.write(f'\n')

        # result = subprocess.run(
        #     ['sbatch', file_name],
        #     capture_output=False, text=False)
        # if result.returncode == 0:
        #     print("Job submitted successfully.")
        # else:
        #     print(f"Job submission failed with error: {result.stderr}")


def dataset_transfer_cm():
    root = 'sat_dataset_trans'
    os.makedirs(root, exist_ok=True)
    graphs = ['lcg', 'vcg']
    models = ['neurosat', 'ggnn']
    fig, axes = plt.subplots(1, 4, figsize=(58, 16))
    models_label = ['NeoruSAT', 'GGNN']
    difficulty_label = ['easy', 'medium']
    for row in range(len(models)):
        for col, difficulty in enumerate(['easy', 'medium']):
            model = models[row]
            graph = graphs[row]
            cm = np.zeros((7, 7)) - 1.
            # for i, ori in enumerate(['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov']):  # 'sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov'
            #     for j, tgt in enumerate(['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov']):
            #         file = f'/home/mila/z/zhaoyu.li/scratch/runs/task=satisfiability_difficulty={difficulty}_dataset={ori}_splits=sat_unsat_label=satisfiability_loss=None/graph={graph}_init_emb=learned_model={model}_n_iterations={ite}_seed={seed}_lr={lr}_weight_decay=1e-08/eval_task=satisfiability_difficulty={difficulty}_ori={ori}_tgt={tgt}_splits=sat_unsat_n_iterations=32_checkpoint=model_best.txt'
            #         if os.path.exists(file) is not True:
            #             cm[i][j] = -1
            #             continue
            #         with open(file, 'r') as f:
            #             lines = f.readlines()
            #
            #         for line in lines:
            #             if 'Overall accuracy' in line:
            #                 line = line.replace(' ', '').split('|')
            #                 cm[i][j] = float(line[2])
            with open(f'{root}/{difficulty}_{model}-{graph}.csv', 'r') as ff:
                data = list(csv.reader(ff))
                for ii in range(len(data)):
                    for jj in range(len(data)):
                        cm[ii][jj] = float(data[ii][jj])

            # cmap = plt.cm.Blues
            # cmap = colors.ListedColormap(cmap(np.linspace(0.3398, 1, cmap.N)))  # 将颜色映射的下半部分变为白色
            axes[row*2+col].imshow(cm, interpolation='nearest', vmin=0.3398, vmax=1., cmap=plt.cm.Blues)
            # plt.title(f'{model}-{graph}')
            # plt.colorbar()
            tick_marks = np.arange(7)
            axes[row*2+col].set_xticks(tick_marks, ['SR', '3-SAT', 'CA', 'PS', r'$k$-Clique', r'$k$-Domset', r'$k$-Vercov'], rotation=45, fontsize=26) # , 'k-vercov'
            axes[row*2+col].set_yticks(tick_marks, ['SR', '3-SAT', 'CA', 'PS', r'$k$-Clique', r'$k$-Domset', r'$k$-Vercov'], fontsize=26)
            axes[row*2+col].tick_params(axis='x', length=0)
            axes[row*2+col].tick_params(axis='y', length=0)
            axes[row*2+col].set_title(f'{models_label[row]} on {difficulty_label[col]} datasets', fontsize=36, y=-0.2)
            # plt.xlabel('Testing dataset')
            # plt.ylabel('Training dataset')
            # print(cm)
            # print()
            for i in range(len(cm)):
                for j in range(len(cm)):
                    axes[row*2+col].text(j, i, f'{cm[i][j] * 100:.2f}', horizontalalignment="center", color="white" if cm[i, j] > 0.65 else "black", fontsize=22)

    plt.savefig(f'{root}/sat_dataset_transfer.pdf', bbox_inches='tight')

def ite_transfer():
    graphs = ['lcg', 'vcg']
    models = ['neurosat', 'ggnn']
    seed = 123
    lr = '0.0001'
    for i in range(len(models)):
        for dataset in ['sr', '3-sat', 'k-clique']:
            model = models[i]
            graph = graphs[i]

            os.makedirs('command', exist_ok=True)
            file_name = f'command/satisfiability_{dataset}_{model}_{graph}_ite_transfer.sh'
            with open(file_name, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write(f'#SBATCH --job-name=eval_{dataset}_ite_transfer\n')
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

                for ori in [8, 16, 32, 64]:  # 'sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov'
                    for tgt in [8, 16, 32, 64]:
                        for difficulty in ['easy', 'medium']:
                            if difficulty == 'medium' and tgt in ['sr', '3-sat']:
                                batch_size = 32
                            else:
                                batch_size = 25
                            f.write(
                                f'python eval_model.py satisfiability /network/scratch/x/xujie.si/satbench/{difficulty}/{dataset}/test/ \\\n')
                            f.write(
                                f'    /network/scratch/x/xujie.si/runs/task\=satisfiability_difficulty\={difficulty}_dataset\={dataset}_splits\=sat_unsat_label\=satisfiability_loss\=None/graph={graph}_init_emb=learned_model={model}_n_iterations={ori}_seed={seed}_lr={lr}_weight_decay=1e-08/checkpoints/model_best.pt \\\n')
                            f.write(f'    --model {model} \\\n')
                            f.write(f'    --graph {graph} \\\n')
                            f.write(f'    --n_iterations {tgt} \\\n')
                            f.write(f'    --batch_size {batch_size} \\\n')
                            f.write(f'    --label satisfiability \\\n')
                            f.write(f'    --test_splits sat unsat \\\n')
                            f.write(f'    --ite_transfer \\\n')
                            f.write(f'    --ori_ite {ori} \n')
                            f.write(f'\n')

            result = subprocess.run(
                ['sbatch', file_name],
                capture_output=False, text=False)
            if result.returncode == 0:
                print("Job submitted successfully.")
            else:
                print(f"Job submission failed with error: {result.stderr}")


def ite_transfer_cm():
    os.makedirs('images', exist_ok=True)
    graphs = ['lcg', 'vcg']
    models = ['neurosat', 'ggnn']
    seed = 123
    lr = '0.0001'
    for _ in range(len(models)):
        for difficulty in ['easy', 'medium']:
            model = models[_]
            graph = graphs[_]
            cm = np.zeros((4, 4)) - 1.
            for dataset in ['sr', '3-sat', 'k-clique']:
                for i, ori in enumerate([8, 16, 32, 64]):  # 'sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov'
                    for j, tgt in enumerate([8, 16, 32, 64]):
                        file = f'/network/scratch/x/xujie.si/runs/task=satisfiability_difficulty={difficulty}_dataset={dataset}_splits=sat_unsat_label=satisfiability_loss=None/graph={graph}_init_emb=learned_model={model}_n_iterations={ori}_seed={seed}_lr={lr}_weight_decay=1e-08/eval_task=satisfiability_difficulty={difficulty}_dataset={dataset}_splits=sat_unsat_ori_ite={ori}_tgt_ite={tgt}_checkpoint=model_best.txt'
                        with open(file, 'r') as f:
                            lines = f.readlines()

                        for line in lines:
                            if 'Overall accuracy' in line:
                                line = line.replace(' ', '').split('|')
                                cm[i][j] = float(line[2])

                plt.figure()
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                # plt.title(f'{model}-{graph}')
                # plt.colorbar()
                tick_marks = np.arange(4)
                plt.xticks(tick_marks, [8, 16, 32, 64])
                plt.yticks(tick_marks, [8, 16, 32, 64])
                plt.tick_params(axis='x', length=0)
                plt.tick_params(axis='y', length=0)
                # plt.xlabel('Testing dataset')
                # plt.ylabel('Training dataset')

                for ii in range(len(cm)):
                    for jj in range(len(cm)):
                        plt.text(jj, ii, f'{cm[ii][jj] * 100:.2f}', horizontalalignment="center",
                                 color="white" if cm[ii, jj] > 0.65 else "black")
                print(cm)
                plt.savefig(f'images/ite_trans_{dataset}_{difficulty}_{model}-{graph}.png', bbox_inches='tight')


def difficulty_transfer_cm():
    os.makedirs('difficulty_transfer', exist_ok=True)
    graphs = ['lcg', 'vcg']
    models = ['neurosat', 'ggnn']
    seed = 123
    lr = '0.0001'
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
                    file = f'/network/scratch/x/xujie.si/runs/task=satisfiability_difficulty={ori}_dataset={dataset}_splits=sat_unsat_label=satisfiability_loss=None/graph={graph}_init_emb=learned_model={model}_n_iterations=32_seed={seed}_lr={lr}_weight_decay=1e-08/eval_task=satisfiability_ori_difficulty={ori}_tgt_difficulty={tgt}_dataset={dataset}_splits=sat_unsat_eval_ite_{32}_checkpoint=model_best.txt'
                    if os.path.exists(file) is not True:
                        continue
                    with open(file, 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        if 'Overall accuracy' in line:
                            line = line.replace(' ', '').split('|')
                            cm[i][j] = float(line[2])
            axes[row][col].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.539, vmax=1)
            axes[row][col].set_xticks(np.arange(3), ['easy', 'medium', 'hard'], fontsize=22)
            axes[row][col].set_yticks(np.arange(2), ['easy', 'medium'], fontsize=22)
            axes[row][col].tick_params(axis='x', length=0)
            axes[row][col].tick_params(axis='y', length=0)
            axes[row][col].set_title(f'{label_dataset[col]}; {label_model[row]}', fontsize=26, y=-0.2)

            for ii in range(len(cm)):
                for jj in range(len(cm[ii])):
                    axes[row][col].text(jj, ii, f'{cm[ii][jj] * 100:.2f}', horizontalalignment="center",
                                        color="white" if cm[ii, jj] > np.mean(cm) else "black", fontsize=22)
            print(dataset, model, cm)
    plt.savefig(f'difficulty_transfer/difficulty_transfer_sat.pdf', bbox_inches='tight')



def difficulty_transfer():
    graphs = ['lcg', 'vcg']
    models = ['neurosat', 'ggnn']
    seed = 123
    lr = '0.0001'

    for i in range(len(models)):
        model = models[i]
        graph = graphs[i]
        file_name = f'command/satisfiability_{model}_{graph}_difficulty_transfer.sh'
        os.makedirs('command', exist_ok=True)
        with open(file_name, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'#SBATCH --job-name=eval_{model}_{graph}_difficulty_transfer\n')
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
                    for tgt in ['hard']:
                        if tgt == 'medium' and dataset == 'k-clique':
                            batch_size = 64
                        else:
                            batch_size = 128
                        train_ite = 32
                        eval_ite = 32
                        f.write(
                            f'python eval_model.py satisfiability /network/scratch/z/zhaoyu.li/satbench/{tgt}/{dataset}/test/ \\\n')
                        f.write(
                            f'    /network/scratch/x/xujie.si/runs/task\=satisfiability_difficulty\={ori}_dataset\={dataset}_splits\=sat_unsat_label\=satisfiability_loss\=None/graph={graph}_init_emb=learned_model={model}_n_iterations={train_ite}_seed={seed}_lr={lr}_weight_decay=1e-08/checkpoints/model_best.pt \\\n')
                        f.write(f'    --model {model} \\\n')
                        f.write(f'    --graph {graph} \\\n')
                        f.write(f'    --n_iterations {eval_ite} \\\n')
                        f.write(f'    --batch_size {batch_size} \\\n')
                        f.write(f'    --label satisfiability \\\n')
                        f.write(f'    --test_splits sat unsat \\\n')
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

# summary_latex()
# eval()
# train_best_ite()
# dataset_transfer()
# ite_transfer()
difficulty_transfer()
# dataset_transfer_cm()
# ite_transfer_cm()
# difficulty_transfer_cm()