python satbench/generators/augmented_formula.py $SCRATCH/satbench/easy/ca/train --splits sat unsat
python satbench/generators/augmented_formula.py $SCRATCH/satbench/easy/ca/valid --splits sat unsat

python satbench/generators/augmented_formula.py $SCRATCH/satbench/medium/ca/train --splits sat unsat
python satbench/generators/augmented_formula.py $SCRATCH/satbench/medium/ca/valid --splits sat unsat

python satbench/generators/augmented_formula.py $SCRATCH/satbench/easy/ps/train --splits sat unsat
python satbench/generators/augmented_formula.py $SCRATCH/satbench/easy/ps/valid --splits sat unsat

python satbench/generators/augmented_formula.py $SCRATCH/satbench/medium/ps/train --splits sat unsat
python satbench/generators/augmented_formula.py $SCRATCH/satbench/medium/ps/valid --splits sat unsat

python satbench/generators/augmented_formula.py $SCRATCH/satbench/easy/k-vercov/train --splits sat unsat
python satbench/generators/augmented_formula.py $SCRATCH/satbench/easy/k-vercov/valid --splits sat unsat

python satbench/generators/augmented_formula.py $SCRATCH/satbench/medium/k-vercov/train --splits sat unsat
python satbench/generators/augmented_formula.py $SCRATCH/satbench/medium/k-vercov/valid --splits sat unsat
