## sr
python satbench/generators/sr.py $SCRATCH/satbench/easy/sr/train 80000 --min_n 10 --max_n 40
python satbench/generators/sr.py $SCRATCH/satbench/easy/sr/valid 10000 --min_n 10 --max_n 40
python satbench/generators/sr.py $SCRATCH/satbench/easy/sr/test 10000 --min_n 10 --max_n 40
python satbench/generators/augmented_formula.py $SCRATCH/satbench/easy/sr/train --splits sat unsat
python satbench/generators/augmented_formula.py $SCRATCH/satbench/easy/sr/valid --splits sat unsat
#
python satbench/generators/sr.py $SCRATCH/satbench/medium/sr/train 80000 --min_n 40 --max_n 200
python satbench/generators/sr.py $SCRATCH/satbench/medium/sr/valid 10000 --min_n 40 --max_n 200
python satbench/generators/sr.py $SCRATCH/satbench/medium/sr/test 10000 --min_n 40 --max_n 200
python satbench/generators/augmented_formula.py $SCRATCH/satbench/medium/sr/train --splits sat unsat
python satbench/generators/augmented_formula.py $SCRATCH/satbench/medium/sr/valid --splits sat unsat
#
python satbench/generators/sr.py $SCRATCH/satbench/hard/sr/test 10000 --min_n 200 --max_n 400
#
## 3-sat
#python satbench/generators/3-sat.py ~/scratch/satbench/easy/3-sat/train 80000 --min_n 10 --max_n 40
#python satbench/generators/3-sat.py ~/scratch/satbench/easy/3-sat/valid 10000 --min_n 10 --max_n 40
#python satbench/generators/3-sat.py ~/scratch/satbench/easy/3-sat/test 10000 --min_n 10 --max_n 40
#python satbench/generators/augmented_formula.py ~/scratch/satbench/easy/3-sat/train --splits sat unsat
#python satbench/generators/augmented_formula.py ~/scratch/satbench/easy/3-sat/valid --splits sat unsat
#
#python satbench/generators/3-sat.py ~/scratch/satbench/medium/3-sat/train 80000 --min_n 40 --max_n 200
#python satbench/generators/3-sat.py ~/scratch/satbench/medium/3-sat/valid 10000 --min_n 40 --max_n 200
#python satbench/generators/3-sat.py ~/scratch/satbench/medium/3-sat/test 10000 --min_n 40 --max_n 200
#python satbench/generators/augmented_formula.py ~/scratch/satbench/medium/3-sat/train --splits sat unsat
#python satbench/generators/augmented_formula.py ~/scratch/satbench/medium/3-sat/valid --splits sat unsat

#python satbench/generators/3-sat.py ~/scratch/satbench/hard/3-sat/test 10000 --min_n 200 --max_n 400
