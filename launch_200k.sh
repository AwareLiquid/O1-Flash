#!/bin/bash
# idempotent launcher: safe to run any number of times
n=$(ps aux | grep -v grep | grep -c 'benchmarks.real_data_bench')
if [ "$n" -gt 0 ]; then
  echo "already running: $n process(es)"
  exit 0
fi
cd /root/o1flash || exit 1
setsid nohup /root/M2/.venv/bin/python -m benchmarks.real_data_bench   --steps 200000 --batch 64 --lr 3e-3 --warmup 500 --min-lr-ratio 0.1   > /root/o1flash/calib_200k_sched.log 2>&1 < /dev/null &
disown
sleep 6
c=$(ps aux | grep -v grep | grep -c 'benchmarks.real_data_bench')
echo "launched, now running: $c"
