# G4SATBench

![overview](assets/overview.png)

This is the official PyTorch implementation of the paper

G4SATBench: Benchmarking and Advancing SAT Solving with Graph Neural Networks</br>
[Zhaoyu Li](https://www.zhaoyu-li.com), Jinpei Guo, and [Xujie Si](https://www.cs.toronto.edu/~six/)</br>

## Installation
We recommand to run the following lines to get started:

```bash
conda create -n g4satbench python=3.9
conda activate g4satbench
bash scripts/install.sh
```

## Datasets
To generate our used SAT datasets, you can use the following scripts:

```bash
# replace ~/g4satbench in the following scripts to your own data directory
bash scripts/gen_data.sh
bash scripts/gen_label.sh

# augmented SAT datasets
bash scripts/gen_aug_data.sh
```

## Benchmarking Evaluation
To train/evaluate a GNN-based SAT sovler, you may run or modify these commands:

```bash
# train NeuroSAT on the easy SR dataset for satisfiability prediction
python train_model.py satisfiability ~/g4satbench/easy/sr/train/ --train_splits sat unsat --valid_dir ~/g4satbench/easy/sr/valid/ --valid_splits sat unsat --label satisfiability --graph lcg --model neurosat --n_iterations 32  --lr 1e-4 --weight_decay 1e-8 --scheduler ReduceLROnPlateau --batch_size 128 --seed 123

# evaluate NeuroSAT on the easy 3-sat dataset for satisfiability prediction
python eval_model.py satisfiability ~/g4satbench/easy/3-sat/test/ runs/task\=satisfiability_difficulty\=easy_dataset\=sr_splits\=sat_unsat/graph=lcg_init_emb=learned_model=neurosat_n_iterations=32_lr=1.0e-4_weight_decay=1.0e-8_seed=123/checkpoints/model_best.pt --test_splits sat unsat --label satisfiability --graph lcg --model neurosat --n_iterations 32 --batch_size 512
    
# train GGNN (VCG) on the easy CA dataset for satisfying assignment prediction with UNS_2 as the training loss
python train_model.py assignment ~/g4satbench/easy/ca/train/ --train_splits sat --valid_dir ~/g4satbench/easy/ca/valid/ --valid_splits sat --loss unsupervised_2 --graph vcg --model ggnn --n_iterations 32  --lr 1e-4 --weight_decay 1e-8 --scheduler ReduceLROnPlateau --batch_size 128 --seed 123

# evaluate GGNN (LCG) on the medium CA dataset for satisfying assignment prediction
python eval_model.py assignment ~/g4satbench/medium/ca/test/ runs/task\=assignment_difficulty\=easy_dataset\=ca_splits\=sat/graph=vcg_init_emb=learned_model=ggnn_n_iterations=32_lr=1e-4_weight_decay=1e-8_seed=123/checkpoints/model_best.pt --test_splits sat --decoding standard --graph lcg --model neurosat --n_iterations 32 --batch_size 512

# train GIN (LCG) on the medium k-clique dataset for unsat-core variable prediction
python train_model.py core_variable ~/g4satbench/medium/k-clique/train/ --train_splits unsat --valid_dir ~/g4satbench/medium/k-clique/valid/ --valid_splits unsat --label core_variable --graph lcg --model gin --n_iterations 32  --lr 1e-4 --weight_decay 1e-8 --scheduler ReduceLROnPlateau --batch_size 128 --seed 123

# evaluate GIN (LCG) on the hard k-clique dataset for unsat-core variable prediction
python eval_model.py satisfiability ~/g4satbench/easy/3-sat/test/ runs/task\=satisfiability_difficulty\=easy_dataset\=sr_splits\=sat_unsat/graph=lcg_init_emb=learned_model=neurosat_n_iterations=32_lr=1e-4_weight_decay=1e-8_seed=123/checkpoints/model_best.pt --test_splits sat unsat --label satisfiability --graph lcg --model neurosat --n_iterations 32 --batch_size 512
```

## Advancing Evaluation
To 
```bash
# train NeuroSAT on the augmented easy SR dataset for satisfiability prediction
python train_model.py satisfiability ~/g4satbench/easy/sr/train/ --train_splits augmented_sat augmented_unsat --valid_dir ~/g4satbench/easy/sr/valid/ --valid_splits augmented_sat augmented_unsat --label satisfiability --graph lcg --model neurosat --n_iterations 32  --lr 1e-4 --weight_decay 1e-8 --scheduler ReduceLROnPlateau --batch_size 128 --seed 123

# evaluate NeuroSAT on the augmented easy sr dataset for satisfiability prediction
python eval_model.py satisfiability ~/g4satbench/easy/sr/test/ runs/task\=satisfiability_difficulty\=easy_dataset\=sr_splits\=augmented_sat_augmented_unsat/graph=lcg_init_emb=learned_model=neurosat_n_iterations=32_lr=1e-4_weight_decay=1e-8_seed=123/checkpoints/model_best.pt --test_splits augmented_sat augmented_unsat --label satisfiability --graph lcg --model neurosat --n_iterations 32 --batch_size 512


```
