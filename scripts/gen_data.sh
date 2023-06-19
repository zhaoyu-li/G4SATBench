# datasets
# sr
python g4satbench/generators/sr.py ~/g4satbench/easy/sr/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_n 10 --max_n 40
python g4satbench/generators/sr.py ~/g4satbench/medium/sr/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_n 40 --max_n 200
python g4satbench/generators/sr.py ~/g4satbench/hard/sr/ --test_instances 10000 --min_n 200 --max_n 400

# 3-sat
python g4satbench/generators/3-sat.py ~/g4satbench/easy/3-sat/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_n 10 --max_n 40
python g4satbench/generators/3-sat.py ~/g4satbench/medium/3-sat/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_n 40 --max_n 200
python g4satbench/generators/3-sat.py ~/g4satbench/hard/3-sat/ --test_instances 10000 --min_n 200 --max_n 300

# ca
python g4satbench/generators/ca.py ~/g4satbench/easy/ca/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_n 10 --max_n 40
python g4satbench/generators/ca.py ~/g4satbench/medium/ca/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_n 40 --max_n 200
python g4satbench/generators/ca.py ~/g4satbench/hard/ca/ --test_instances 10000 --min_n 200 --max_n 400

# ps
python g4satbench/generators/ps.py ~/g4satbench/easy/ps/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_n 10 --max_n 40
python g4satbench/generators/ps.py ~/g4satbench/medium/ps/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_n 40 --max_n 200
python g4satbench/generators/ps.py ~/g4satbench/hard/ps/ --test_instances 10000 --min_n 200 --max_n 300

# k-clique
python g4satbench/generators/k-clique.py ~/g4satbench/easy/k-clique/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 5 --max_v 15  --min_k 3 --max_k 4
python g4satbench/generators/k-clique.py ~/g4satbench/medium/k-clique/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 15 --max_v 20  --min_k 3 --max_k 5
python g4satbench/generators/k-clique.py ~/g4satbench/hard/k-clique/ --test_instances 10000 --min_v 20 --max_v 25  --min_k 4 --max_k 6

# k-domset
python g4satbench/generators/k-domset.py ~/g4satbench/easy/k-domset/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 5 --max_v 15  --min_k 2 --max_k 3
python g4satbench/generators/k-domset.py ~/g4satbench/medium/k-domset/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 15 --max_v 20  --min_k 3 --max_k 5
python g4satbench/generators/k-domset.py ~/g4satbench/hard/k-domset/ --test_instances 10000 --min_v 20 --max_v 25  --min_k 4 --max_k 6

# k-vercov
python g4satbench/generators/k-vercov.py ~/g4satbench/easy/k-vercov/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 5 --max_v 15  --min_k 3 --max_k 5
python g4satbench/generators/k-vercov.py ~/g4satbench/medium/k-vercov/ --train_instances 80000 --valid_instances 10000 --test_instances 10000 --min_v 10 --max_v 20  --min_k 6 --max_k 8
python g4satbench/generators/k-vercov.py ~/g4satbench/hard/k-vercov/ --test_instances 10000 --min_v 15 --max_v 25  --min_k 9 --max_k 10