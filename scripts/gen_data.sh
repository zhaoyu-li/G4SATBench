# datasets
# sr
python satbench/generators/sr.py ~/satbench/easy/sr/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_n 10 --max_n 40
python satbench/generators/sr.py ~/satbench/medium/sr/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_n 40 --max_n 200
python satbench/generators/sr.py ~/satbench/hard/sr/ --test_instances 10000 --min_n 200 --max_n 400

# 3-sat
python satbench/generators/3-sat.py ~/satbench/easy/3-sat/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_n 10 --max_n 40
python satbench/generators/3-sat.py ~/satbench/medium/3-sat/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_n 40 --max_n 200
python satbench/generators/3-sat.py ~/satbench/hard/3-sat/ --test_instances 10000 --min_n 200 --max_n 300

# ca
python satbench/generators/ca.py ~/satbench/easy/ca/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_n 10 --max_n 40
python satbench/generators/ca.py ~/satbench/medium/ca/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_n 40 --max_n 200
python satbench/generators/ca.py ~/satbench/hard/ca/ --test_instances 10000 --min_n 200 --max_n 400

# ps
python satbench/generators/ps.py ~/satbench/easy/ps/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_n 10 --max_n 40
python satbench/generators/ps.py ~/satbench/medium/ps/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_n 40 --max_n 200
python satbench/generators/ps.py ~/satbench/hard/ps/ --test_instances 10000 --min_n 200 --max_n 300

# k-clique
python satbench/generators/k-clique.py ~/satbench/easy/k-clique/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 5 --max_v 15  --min_k 3 --max_k 4
python satbench/generators/k-clique.py ~/satbench/medium/k-clique/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 15 --max_v 20  --min_k 3 --max_k 5
python satbench/generators/k-clique.py ~/satbench/hard/k-clique/ --test_instances 10000 --min_v 20 --max_v 25  --min_k 4 --max_k 6

# k-domset
python satbench/generators/k-domset.py ~/satbench/easy/k-domset/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 5 --max_v 15  --min_k 2 --max_k 3
python satbench/generators/k-domset.py ~/satbench/medium/k-domset/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 15 --max_v 20  --min_k 3 --max_k 5
python satbench/generators/k-domset.py ~/satbench/hard/k-domset/ --test_instances 10000 --min_v 20 --max_v 25  --min_k 4 --max_k 6

# k-vercov
python satbench/generators/k-vercov.py ~/satbench/easy/k-vercov/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 5 --max_v 15  --min_k 3 --max_k 5
python satbench/generators/k-vercov.py ~/satbench/medium/k-vercov/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 10 --max_v 20  --min_k 6 --max_k 8
python satbench/generators/k-vercov.py ~/satbench/hard/k-vercov/ --test_instances 10000 --min_v 15 --max_v 25  --min_k 9 --max_k 10