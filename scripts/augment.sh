python satbench/generators/augmented_formula.py $SCRATCH/satbench/easy/sr/train --splits sat unsat
python satbench/generators/augmented_formula.py $SCRATCH/satbench/easy/sr/valid --splits sat unsat

python satbench/generators/augmented_formula.py $SCRATCH/satbench/medium/sr/train --splits sat unsat
python satbench/generators/augmented_formula.py $SCRATCH/satbench/medium/sr/valid --splits sat unsat

python satbench/generators/augmented_formula.py $SCRATCH/satbench/easy/3-sat/train --splits sat unsat
python satbench/generators/augmented_formula.py $SCRATCH/satbench/easy/3-sat/valid --splits sat unsat

python satbench/generators/augmented_formula.py $SCRATCH/satbench/medium/3-sat/train --splits sat unsat
python satbench/generators/augmented_formula.py $SCRATCH/satbench/medium/3-sat/valid --splits sat unsat
