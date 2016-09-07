python3 Exp10.py > Exp10.logs 2>&1
for i in `seq 1 9`; do
	python3 Exp10a.py >> Exp10.logs 2>&1
done
