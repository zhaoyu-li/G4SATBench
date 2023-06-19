# augmented datasets
# sr
python satbench/generators/augmented_formula.py ~/satbench/easy/sr/train --splits sat unsat
python satbench/generators/augmented_formula.py ~/satbench/easy/sr/valid --splits sat unsat
python satbench/generators/augmented_formula.py ~/satbench/easy/sr/test --splits sat unsat

python satbench/generators/augmented_formula.py ~/satbench/medium/sr/train --splits sat unsat
python satbench/generators/augmented_formula.py ~/satbench/medium/sr/valid --splits sat unsat
python satbench/generators/augmented_formula.py ~/satbench/medium/sr/test --splits sat unsat

# 3-sat
python satbench/generators/augmented_formula.py ~/satbench/easy/3-sat/train --splits sat unsat
python satbench/generators/augmented_formula.py ~/satbench/easy/3-sat/valid --splits sat unsat
python satbench/generators/augmented_formula.py ~/satbench/easy/3-sat/test --splits sat unsat

python satbench/generators/augmented_formula.py ~/satbench/medium/3-sat/train --splits sat unsat
python satbench/generators/augmented_formula.py ~/satbench/medium/3-sat/valid --splits sat unsat
python satbench/generators/augmented_formula.py ~/satbench/medium/3-sat/test --splits sat unsat
