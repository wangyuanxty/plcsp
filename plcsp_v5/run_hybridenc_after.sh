#!/bin/bash
# Wait for the 10x6 RS queue, then run the missing encoder cell
# (job GIN + machine ST, 6x6, 3 seeds).
cd /root/plcsp_v5 || exit 1
LOG=queue_logs/queue_local_10x6.log
while ! grep -q "queue done" "$LOG" 2>/dev/null; do
    sleep 60
done
setsid nohup /root/venv/bin/python run_queue.py \
    --config experiment-queue-hybrid-encoder.json \
    > queue_logs/queue_hybrid_encoder.log 2>&1 < /dev/null &
echo "hybrid encoder queue launched at $(date -Is)"
