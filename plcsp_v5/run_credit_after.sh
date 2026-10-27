#!/bin/bash
# Wait for the running non-RS queues to drain, then start the top-level
# credit-assignment experiments.  Launch with:
#   setsid nohup bash run_credit_after.sh > queue_logs/run_credit_after.log 2>&1 < /dev/null &
cd /root/plcsp_v5 || exit 1
while pgrep -f 'python .*Run\.py|python .*run_queue\.py' > /dev/null; do
    sleep 60
done
setsid nohup /root/venv/bin/python run_queue.py \
    --config experiment-queue-credit.json \
    > queue_logs/queue_credit.log 2>&1 < /dev/null &
echo "credit queue launched at $(date -Is)"
