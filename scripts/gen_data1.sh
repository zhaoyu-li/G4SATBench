# ca
#python satbench/generators/ca.py $SCRATCH/satbench/easy/ca/train 80000 --min_n 10 --max_n 40
#python satbench/generators/ca.py $SCRATCH/satbench/easy/ca/valid 10000 --min_n 10 --max_n 40
#python satbench/generators/ca.py $SCRATCH/satbench/easy/ca/test 10000 --min_n 10 --max_n 40
#python satbench/generators/augmented_formula.py $SCRATCH/satbench/easy/ca/train --splits sat unsat
#python satbench/generators/augmented_formula.py $SCRATCH/satbench/easy/ca/valid --splits sat unsat
#
#python satbench/generators/ca.py $SCRATCH/satbench/medium/ca/train 80000 --min_n 40 --max_n 200
#python satbench/generators/ca.py $SCRATCH/satbench/medium/ca/valid 10000 --min_n 40 --max_n 200
#python satbench/generators/ca.py $SCRATCH/satbench/medium/ca/test 10000 --min_n 40 --max_n 200
#python satbench/generators/augmented_formula.py $SCRATCH/satbench/medium/ca/train --splits sat unsat
#python satbench/generators/augmented_formula.py $SCRATCH/satbench/medium/ca/valid --splits sat unsat
#
#python satbench/generators/ca.py $SCRATCH/satbench/hard/ca/test 10000 --min_n 200 --max_n 400

# ps
#python satbench/generators/ps.py $SCRATCH/satbench/easy/ps/train 80000 --min_n 10 --max_n 40
#python satbench/generators/ps.py $SCRATCH/satbench/easy/ps/valid 10000 --min_n 10 --max_n 40
#python satbench/generators/ps.py $SCRATCH/satbench/easy/ps/test 10000 --min_n 10 --max_n 40
#python satbench/generators/augmented_formula.py $SCRATCH/satbench/easy/ps/train --splits sat unsat
#python satbench/generators/augmented_formula.py $SCRATCH/satbench/easy/ps/valid --splits sat unsat

python satbench/generators/ps.py $SCRATCH/satbench/medium/ps/train 80000 --min_n 40 --max_n 200
python satbench/generators/ps.py $SCRATCH/satbench/medium/ps/valid 10000 --min_n 40 --max_n 200
python satbench/generators/ps.py $SCRATCH/satbench/medium/ps/test 10000 --min_n 40 --max_n 200
python satbench/generators/augmented_formula.py $SCRATCH/satbench/medium/ps/train --splits sat unsat
python satbench/generators/augmented_formula.py $SCRATCH/satbench/medium/ps/valid --splits sat unsat

python satbench/generators/ps.py $SCRATCH/satbench/hard/ps/test 10000 --min_n 200 --max_n 400
