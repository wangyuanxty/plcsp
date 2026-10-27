#!/bin/bash
# Wait for the hybrid queue to finish, then run the 10x6 all-GNN (e2e) seeds 2/3.
cd /root/plcsp_v5 || exit 1
LOG=queue_logs/queue_credit_hybrid.log
while ! grep -q "queue done" "$LOG" 2>/dev/null; do
    sleep 60
done
setsid nohup /root/venv/bin/python run_queue.py \
    --config experiment-queue-st-e2e-10x6-rest.json \
    > queue_logs/queue_st_e2e_10x6.log 2>&1 < /dev/null &
echo "st e2e 10x6 queue launched at $(date -Is)"
