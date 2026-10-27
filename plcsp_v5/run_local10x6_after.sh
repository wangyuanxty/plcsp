#!/bin/bash
# Wait for the e2e 10x6 queue to finish, then run the 10x6 main model
# (RS + local credit, 3 seeds).
cd /root/plcsp_v5 || exit 1
LOG=queue_logs/queue_st_e2e_10x6.log
while ! grep -q "queue done" "$LOG" 2>/dev/null; do
    sleep 60
done
setsid nohup /root/venv/bin/python run_queue.py \
    --config experiment-queue-local-10x6.json \
    > queue_logs/queue_local_10x6.log 2>&1 < /dev/null &
echo "local 10x6 queue launched at $(date -Is)"
