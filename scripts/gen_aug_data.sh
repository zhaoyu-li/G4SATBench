# augmented datasets
# sr
python g4satbench/generators/augmented_formula.py ~/g4satbench/easy/sr/train --splits sat unsat
python g4satbench/generators/augmented_formula.py ~/g4satbench/easy/sr/valid --splits sat unsat
python g4satbench/generators/augmented_formula.py ~/g4satbench/easy/sr/test --splits sat unsat

python g4satbench/generators/augmented_formula.py ~/g4satbench/medium/sr/train --splits sat unsat
python g4satbench/generators/augmented_formula.py ~/g4satbench/medium/sr/valid --splits sat unsat
python g4satbench/generators/augmented_formula.py ~/g4satbench/medium/sr/test --splits sat unsat

# 3-sat
python g4satbench/generators/augmented_formula.py ~/g4satbench/easy/3-sat/train --splits sat unsat
python g4satbench/generators/augmented_formula.py ~/g4satbench/easy/3-sat/valid --splits sat unsat
python g4satbench/generators/augmented_formula.py ~/g4satbench/easy/3-sat/test --splits sat unsat

python g4satbench/generators/augmented_formula.py ~/g4satbench/medium/3-sat/train --splits sat unsat
python g4satbench/generators/augmented_formula.py ~/g4satbench/medium/3-sat/valid --splits sat unsat
python g4satbench/generators/augmented_formula.py ~/g4satbench/medium/3-sat/test --splits sat unsat
