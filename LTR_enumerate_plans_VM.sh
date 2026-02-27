#!/bin/bash

nohup python3 -u  LTR_enumerate_plans_VM.py --emd $1 --tq $2  --db $3 --mn $4  --iter $5  >./logs/LTR_enum_$1_tq_$2_db_$3_mn_$4_iter_$5.log 2>&1 &

