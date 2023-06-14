import os

# tgt_dir = 'transfer_dir3/'
# os.makedirs(tgt_dir, exist_ok=True)
# for task in ['core_variable']: # 'satisfiability', 'core_variable', 'assignment'
#     if task == 'satisfiability':
#         split = 'sat_unsat'
#     elif task == 'assignment':
#         split = 'sat'
#     else:
#         split = 'unsat'
#     for difficulty in ['easy', 'medium']: # 'easy',
#         for loss in ['supervised', 'unsupervised']: # ,'unsupervisedv2'
#             for dataset in ['k-vercov']: # 'sr', '3-sat', 'ca', 'ps', 'k-vercov', 'k-clique', 'k-domset'
#                 if task in ['satisfiability',  'core_variable']:
#                     loss = 'None'
#                 if task == 'assignment' and loss in ['unsupervised', 'unsupervisedv2']:
#                     label = 'None'
#                 else:
#                     label = task
#                 dir = f'/network/scratch/x/xujie.si/runs/task={task}_difficulty={difficulty}_dataset={dataset}_splits={split}_label={label}_loss={loss}'
#                 content = os.listdir(dir)
#                 for sub_dir in content:
#                     if sub_dir not in [
#                                        'graph=lcg_init_emb=learned_model=neurosat_n_iterations=32_seed=123_lr=0.0001_weight_decay=1e-08',
#                                        'graph=vcg_init_emb=learned_model=ggnn_n_iterations=32_seed=123_lr=0.0001_weight_decay=1e-08',
#                                        'graph=lcg_init_emb=learned_model=neurosat_n_iterations=32_seed=234_lr=0.0001_weight_decay=1e-08',
#                                        'graph=vcg_init_emb=learned_model=ggnn_n_iterations=32_seed=234_lr=0.0001_weight_decay=1e-08',
#                                        'graph=lcg_init_emb=learned_model=neurosat_n_iterations=32_seed=345_lr=0.0001_weight_decay=1e-08',
#                                        'graph=vcg_init_emb=learned_model=ggnn_n_iterations=32_seed=345_lr=0.0001_weight_decay=1e-08'
#                                        ]:
#                         continue
#                     # tgt_sub_dir = os.path.join(tgt_dir, f'task={task}_difficulty={difficulty}_dataset={dataset}_splits={split}_label={label}_loss={loss}', sub_dir)
#                     # file = os.path.join(dir, sub_dir, f'eval_task={task}_difficulty={difficulty}_dataset={dataset}_splits=sat_decoding=standard_n_iterations=32_checkpoint=model_best.txt')
#                     # os.system(f'cp {file} {tgt_sub_dir}')
#                     # os.system(f'ls {dir}/{sub_dir}/checkpoints')
#                     # file = os.path.join(dir, sub_dir, 'checkpoints/model_best.pt')
#                     # if os.path.exists(file) is not True:
#                     #     continue
#                     # os.makedirs(tgt_sub_dir, exist_ok=True)
#
#                     # os.makedirs(f'{tgt_sub_dir}/checkpoints', exist_ok=True)
#                     # os.system(f'cp {file} {tgt_sub_dir}/checkpoints')
#
#                     # os.system(f'ls {dir}/{sub_dir}')
#                     if os.path.exists(f'{dir}/{sub_dir}/model_best.pt'):
#                         os.system(f'mkdir {dir}/{sub_dir}/checkpoints')
#                         os.system(f'mv {dir}/{sub_dir}/model_best.pt {dir}/{sub_dir}/checkpoints')
                     # os.system(f'mkdir {os.path.join(dir, sub_dir, "checkpoints")}')
                    # os.system(f'mv {file} {os.path.join(dir, sub_dir, "checkpoints")}')

# for difficulty in ['easy', 'medium']:
#     for dataset in ['k-clique', 'k-domset', 'sr', '3-sat', 'ca', 'ps']:
#         dir = f'/network/scratch/z/zhaoyu.li/runs/'
#         tgt_dir = os.path.join(dir, f"task=satisfiability_difficulty={difficulty}_dataset={dataset}_splits=sat_unsat_label=satisfiability_loss=None")
#         ori_dir = os.path.join(dir, f"task=satisfiability_difficulty={difficulty}_dataset={dataset}_splits=sat_label=satisfiability_loss=None")
#         if os.path.exists(ori_dir) is True:
#             os.system(f'mv {ori_dir} {tgt_dir}')

for difficulty in ['easy']:
    root =  f'/network/scratch/z/zhaoyu.li/runs/task=assignment_difficulty=easy_dataset=k-vercov_splits=sat_unsat_label=None_loss=unsupervised'
    subs = os.listdir(root)
    for sub in subs:
        eval_file = os.path.join(root, sub, f'eval_task=assignment_difficulty={difficulty}_dataset=k-vercov_splits=sat_decoding=standard_n_iterations=32_checkpoint=model_best.txt')
        ckpt = os.path.join(root, sub, f'checkpoints/model_best.pt')
        tgt_dir = f'transfer/task=assignment_difficulty=easy_dataset=k-vercov_splits=sat_unsat_label=None_loss=unsupervised/{sub}'
        os.makedirs(tgt_dir, exist_ok=True)
        if os.path.exists(eval_file):
            os.system(f'cp {eval_file} {tgt_dir}')
        os.makedirs(f'{tgt_dir}/checkpoints', exist_ok=True)
        os.system(f'cp {ckpt} {tgt_dir}/checkpoints')


