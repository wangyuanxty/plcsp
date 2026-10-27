#!/bin/bash
# Wait for the hybrid encoder queue, then run E2E + AGV (3 seeds).
cd /root/plcsp_v5 || exit 1
LOG=queue_logs/queue_hybrid_encoder.log
while ! grep -q "queue done" "$LOG" 2>/dev/null; do
    sleep 60
done
setsid nohup /root/venv/bin/python run_queue.py \
    --config experiment-queue-e2e-agv.json \
    > queue_logs/queue_e2e_agv.log 2>&1 < /dev/null &
echo "e2e agv queue launched at $(date -Is)"
