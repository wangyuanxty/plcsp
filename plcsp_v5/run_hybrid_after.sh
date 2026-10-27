#!/bin/bash
# Wait for the credit and 10x6 PPO queues to drain, then start hybrid.
cd /root/plcsp_v5 || exit 1
while pgrep -f 'python .*Run\.py|python .*run_queue\.py' > /dev/null; do
    sleep 60
done
setsid nohup /root/venv/bin/python run_queue.py \
    --config experiment-queue-credit-hybrid.json \
    > queue_logs/queue_credit_hybrid.log 2>&1 < /dev/null &
echo "hybrid queue launched at $(date -Is)"
