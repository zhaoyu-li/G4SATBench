import os
import subprocess
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import csv


def fail_case():
    models = ['neurosat', 'neurosat', 'ggnn', 'ggnn', 'ggnn']
    graphs = ['lcg', 'lcg', 'lcg', 'vcg', 'vcg']
    datasets = ['ca', 'k-clique', 'ca', 'ca', 'k-clique']

    for i in range(len(models)):
        os.makedirs('command', exist_ok=True)
        seed = 123
        ite = 32
        difficulty = 'medium'
        model = models[i]
        graph = graphs[i]
        dataset = datasets[i]
        file_name = f'command/core_variable_train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{seed}.sh'
        with open(file_name, 'w') as f:
            lr = 0.0001
            batch_size = 32
            f.write('#!/bin/bash\n')
            f.write(
                f'#SBATCH --job-name=train_core_{model}_{ite}_{graph}_{dataset}_{difficulty}_{seed}\n')
            f.write('#SBATCH --output=/dev/null\n')
            f.write('#SBATCH --ntasks=1\n')
            f.write('#SBATCH --time=2-23:00:00\n')
            f.write('#SBATCH --gres=gpu:rtx8000:1\n')
            f.write('#SBATCH --mem=16G\n')
            f.write('#SBATCH --cpus-per-task=8\n')
            f.write('\n')
            f.write('module load anaconda/3\n')
            f.write('conda activate satbench\n')
            f.write('\n')
            f.write(
                f'python train_model.py core_variable $SCRATCH/satbench/{difficulty}/{dataset}/train/ \\\n')
            f.write('    --train_splits unsat \\\n')
            f.write(f'    --valid_dir $SCRATCH/satbench/{difficulty}/{dataset}/valid/ \\\n')
            f.write('    --valid_splits unsat \\\n')
            f.write('    --label core_variable \\\n')
            f.write('    --scheduler ReduceLROnPlateau \\\n')
            f.write(f'    --lr {lr} \\\n')
            f.write(f'    --n_iterations {ite} \\\n')
            f.write('    --weight_decay 1.e-8 \\\n')
            f.write(f'    --model {model} \\\n')
            f.write(f'    --graph {graph} \\\n')
            f.write(f'    --epochs 100 \\\n')
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
        for dataset in ['sr','3-sat','ca','ps','k-clique', 'k-domset']: # 'ca', 'ps', 'k-clique',  'k-domset', 'k-color'
            for model in ['gcn', 'gin', 'ggnn', 'neurosat']: # 'gcn', 'gin', 'ggnn', 'neurosat'
                for ite in [32]:
                    for graph in ['lcg', 'vcg']: # 'lcg', 'vcg'
                        for difficulty in ['medium']:
                            if model == 'neurosat' and graph == 'vcg':
                                continue
                            file_name = f'command/core_variable_train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{seed}.sh'
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
                                f.write(f'#SBATCH --job-name=train_{model}_{ite}_{graph}_{dataset}_{difficulty}_{seed}\n')
                                f.write('#SBATCH --output=/dev/null\n')
                                f.write('#SBATCH --ntasks=1\n')
                                f.write('#SBATCH --time=2-23:00:00\n')
                                f.write('#SBATCH --gres=gpu:rtx8000:1\n')
                                f.write('#SBATCH --mem=16G\n')
                                f.write('#SBATCH --cpus-per-task=8\n')
                                f.write('\n')
                                f.write('module load anaconda/3\n')
                                f.write('conda activate satbench\n')
                                f.write('\n')
                                f.write(f'python train_model.py core_variable $SCRATCH/satbench/{difficulty}/{dataset}/train/ \\\n')
                                f.write('    --train_splits unsat \\\n')
                                f.write(f'    --valid_dir $SCRATCH/satbench/{difficulty}/{dataset}/valid/ \\\n')
                                f.write('    --valid_splits unsat \\\n')
                                f.write('    --label core_variable \\\n')
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



def eval():
    file_name = f'command/core_eval.sh'
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
            for dataset in ['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset']: # 'k-vercov'
                for model in ['gcn', 'gin', 'ggnn', 'neurosat']:
                    for ite in [32]:
                        for graph in ['lcg', 'vcg']:
                            for difficulty in ['medium']: # 'easy',
                                if model == 'neurosat' and graph == 'vcg':
                                    continue
                                lr = '0.0001'
                                if dataset in ['k-domset', 'k-clique'] and difficulty == 'medium':
                                    batch_size = 64
                                else:
                                    batch_size = 128
                                f.write(
                                    f'python eval_model.py core_variable /network/scratch/x/xujie.si/satbench/{difficulty}/{dataset}/test/ \\\n')
                                f.write(f'    /network/scratch/x/xujie.si/runs/task\=core_variable_difficulty\={difficulty}_dataset\={dataset}_splits\=unsat_label\=core_variable_loss\=None/graph={graph}_init_emb=learned_model={model}_n_iterations={ite}_seed={seed}_lr={lr}_weight_decay=1e-08/checkpoints/model_best.pt \\\n')
                                f.write(f'    --model {model} \\\n')
                                f.write(f'    --graph {graph} \\\n')
                                f.write(f'    --n_iterations {ite} \\\n')
                                f.write(f'    --batch_size {batch_size} \\\n')
                                f.write(f'    --label core_variable \\\n')
                                f.write(f'    --test_splits unsat\n')
                                f.write(f'\n')

    result = subprocess.run(
        ['sbatch', f'command/core_eval.sh'],
        capture_output=False, text=False)
    if result.returncode == 0:
        print("Job submitted successfully.")
    else:
        print(f"Job submission failed with error: {result.stderr}")


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
        file_name = f'command/core_{model}_{graph}_dataset_transfer.sh'
        with open(file_name, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'#SBATCH --job-name=eval_core_{model}_{graph}_dataset_transfer\n')
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
            for ori in ['k-vercov']:  # 'sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov'
                for tgt in ['k-vercov']:
                    for difficulty in ['easy', 'medium']:
                        if difficulty == 'medium' and tgt in ['ca', 'k-clique']:
                            batch_size =50
                        else:
                            batch_size = 128
                        user = 'z/zhaoyu.li'
                        f.write(f'python eval_model.py core_variable /network/scratch/{user}/satbench/{difficulty}/{tgt}/test/ \\\n')
                        f.write(f'    /network/scratch/{user}/runs/task\=core_variable_difficulty\={difficulty}_dataset\={ori}_splits\=unsat_label\=core_variable_loss\=None/graph={graph}_init_emb=learned_model={model}_n_iterations={ite}_seed={seed}_lr={lr}_weight_decay=1e-08/checkpoints/model_best.pt \\\n')
                        f.write(f'    --model {model} \\\n')
                        f.write(f'    --graph {graph} \\\n')
                        f.write(f'    --n_iterations {ite} \\\n')
                        f.write(f'    --batch_size {batch_size} \\\n')
                        f.write(f'    --label core_variable \\\n')
                        f.write(f'    --test_splits unsat \\\n')
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
    root = 'core_dataset_trans'
    os.makedirs(root, exist_ok=True)
    graphs = ['lcg', 'vcg']
    models = ['neurosat', 'ggnn']
    for _ in range(len(models)):
        for difficulty in ['easy', 'medium']:
            model = models[_]
            graph = graphs[_]
            cm = np.zeros((7, 7)) - 1.
            # for i, ori in enumerate(['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset',
            #                          'k-vercov']):  # 'sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov'
            #     for j, tgt in enumerate(['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov']):
                    # file = f'/home/mila/x/xujie.si/scratch/runs/task=core_variable_difficulty={difficulty}_dataset={ori}_splits=unsat_label=core_variable_loss=None/graph={graph}_init_emb=learned_model={model}_n_iterations={ite}_seed={seed}_lr={lr}_weight_decay=1e-08/eval_task=core_variable_difficulty={difficulty}_ori={ori}_tgt={tgt}_splits=unsat_n_iterations=32_checkpoint=model_best.txt'
                    # if os.path.exists(file) is not True:
                    #     cm[i][j] = -1
                    #     continue
                    # with open(file, 'r') as f:
                    #     lines = f.readlines()
                    #
                    # for line in lines:
                    #     if 'Overall accuracy' in line:
                    #         line = line.replace(' ', '').split('|')
                    #         cm[i][j] = float(line[2])

            with open(f'{root}/{difficulty}_{model}-{graph}.csv', 'r') as ff:
                data = list(csv.reader(ff))
                for ii in range(len(data)):
                    for jj in range(len(data)):
                        cm[ii][jj] = float(data[ii][jj])
            plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0., vmax=1.)

            tick_marks = np.arange(7)
            plt.xticks(tick_marks, ['SR', '3-SAT', 'CA', 'PS', r'$k$-Clique', r'$k$-Domset', r'$k$-Vercov'])  # , 'k-vercov'
            plt.yticks(tick_marks, ['SR', '3-SAT', 'CA', 'PS', r'$k$-Clique', r'$k$-Domset', r'$k$-Vercov'])
            plt.tick_params(axis='x', length=0)
            plt.tick_params(axis='y', length=0)
            plt.xticks(rotation=45)
            print(np.around(cm, decimals=2))
            print()
            for i in range(len(cm)):
                for j in range(len(cm)):
                    plt.text(j, i, f'{cm[i][j] * 100:.2f}', horizontalalignment="center",
                             color="white" if cm[i, j] > 0.65 else "black")

            plt.savefig(f'{root}/core_{difficulty}_{model}-{graph}_dataset_transfer.pdf', bbox_inches='tight')


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
                            dir = f'/network/scratch/x/xujie.si/runs/task=core_variable_difficulty={difficulty}_dataset={dataset}_splits=unsat_label=core_variable_loss=None/graph={graph}_init_emb=learned_model={model}_n_iterations={ite}_seed={seed}_lr={lr}_weight_decay=1e-08/'
                            file_name = f'eval_task=core_variable_difficulty={difficulty}_dataset={dataset}_splits=unsat_n_iterations={ite}_checkpoint=model_best.txt'

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

    file_name = f'results/core_eval.csv'

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
    csv_name = f'results/core_eval.csv'
    latex_name = f'results/core_eval.txt'
    if os.path.exists(latex_name):
        os.system(f'rm {latex_name}')
    with open(csv_name, 'r') as f:
        for str in f.readlines():
            with open(latex_name, 'a') as ff:
                str = str.replace(',', ' & ')
                ff.write(str)


def difficulty_transfer():
    graphs = ['vcg', 'lcg']
    models = ['ggnn', 'neurosat']
    seed = 123
    lr = '0.0001'

    for i in range(len(models)):
        model = models[i]
        graph = graphs[i]
        file_name = f'command/core_{model}_{graph}_difficulty_transfer.sh'
        os.makedirs('command', exist_ok=True)
        with open(file_name, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'#SBATCH --job-name=core_eval_{model}_{graph}_difficulty_transfer\n')
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
                        user = 'x/xujie.si'
                        batch_size = 50
                        train_ite = 32
                        eval_ite = 32
                        f.write(
                            f'python eval_model.py core_variable /network/scratch/{user}/satbench/{tgt}/{dataset}/test/ \\\n')
                        f.write(
                            f'    /network/scratch/{user}/runs/task\=core_variable_difficulty\={ori}_dataset\={dataset}_splits\=unsat_label\=core_variable_loss\=None/graph={graph}_init_emb=learned_model={model}_n_iterations={train_ite}_seed={seed}_lr={lr}_weight_decay=1e-08/checkpoints/model_best.pt \\\n')
                        f.write(f'    --model {model} \\\n')
                        f.write(f'    --graph {graph} \\\n')
                        f.write(f'    --n_iterations {eval_ite} \\\n')
                        f.write(f'    --batch_size {batch_size} \\\n')
                        f.write(f'    --label core_variable \\\n')
                        f.write(f'    --test_splits unsat \\\n')
                        f.write(f'    --difficulty_transfer \\\n')
                        f.write(f'    --ori_difficulty {ori} \n')
                        f.write(f'\n')

        # result = subprocess.run(
        #     ['sbatch', file_name],
        #     capture_output=False, text=False)
        # if result.returncode == 0:
        #     print("Job submitted successfully.")
        # else:
        #     print(f"Job submission failed with error: {result.stderr}")


def difficulty_transfer_cm():
    os.makedirs('core_difficulty_transfer', exist_ok=True)
    graphs = ['lcg', 'vcg']
    models = ['neurosat', 'ggnn']
    seed = 123
    lr = '0.0001'
    fig, axes = plt.subplots(2, 7, figsize=(21, 16))
    label_model = ['NeuroSAT', 'GGNN']
    label_dataset = ['SR', '3-SAT', 'CA', 'PS', r'$k$-Clique', r'$k$-Domset', r'$k$-Vercov']
    for row in range(len(models)):
        model = models[row]
        graph = graphs[row]
        for col, dataset in enumerate(['sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov']):
            cm = np.zeros((2, 3)) - 1.
            for i, ori in enumerate(['easy', 'medium']):  # 'sr', '3-sat', 'ca', 'ps', 'k-clique', 'k-domset', 'k-vercov'
                for j, tgt in enumerate(['easy', 'medium', 'hard']):
                    file = f'/network/scratch/x/xujie.si/runs/task=core_variable_difficulty={ori}_dataset={dataset}_splits=unsat_label=core_variable_loss=None/graph={graph}_init_emb=learned_model={model}_n_iterations=32_seed={seed}_lr={lr}_weight_decay=1e-08/eval_task=core_variable_ori_difficulty={ori}_tgt_difficulty={tgt}_dataset={dataset}_splits=unsat_eval_ite_32_checkpoint=model_best.txt'
                    if os.path.exists(file) is not True:
                        continue
                    with open(file, 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        if 'Overall accuracy' in line:
                            line = line.replace(' ', '').split('|')
                            cm[i][j] = float(line[2])
            # axes[row][col].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.539, vmax=1)
            # axes[row][col].set_xticks(np.arange(3), ['easy', 'medium', 'hard'], fontsize=22)
            # axes[row][col].set_yticks(np.arange(2), ['easy', 'medium'], fontsize=22)
            # axes[row][col].tick_params(axis='x', length=0)
            # axes[row][col].tick_params(axis='y', length=0)
            # axes[row][col].set_title(f'{label_dataset[col]}; {label_model[row]}', fontsize=26, y=-0.2)
            #
            # for ii in range(len(cm)):
            #     for jj in range(len(cm[ii])):
            #         axes[row][col].text(jj, ii, f'{cm[ii][jj] * 100:.2f}', horizontalalignment="center",
            #                             color="white" if cm[ii, jj] > np.mean(cm) else "black", fontsize=22)

            print(dataset, model, cm)
    # plt.savefig(f'difficulty_transfer/difficulty_transfer_sat.pdf', bbox_inches='tight')

# eval()
# summary_latex()
# fail_case()
# summary_csv()
# dataset_transfer()
# dataset_transfer_cm()
difficulty_transfer()
# difficulty_transfer_cm()