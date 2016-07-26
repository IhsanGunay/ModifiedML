#! /bin/bash

python3 Exp6.py > Exp6.logs 2>&1

for i in `seq 1 9`; do
	python3 Exp6a.py >> Exp6.logs 2>&1
done
