#! /bin/bash

python3 Exp9.py 2 > Exp9.logs 2>&1
python3 Exp9.py 3 >> Exp9.logs 2>&1
python3 Exp9.py 5 >> Exp9.logs 2>&1
python3 Exp9.py 10 >> Exp9.logs 2>&1
