#!/bin/bash
# Sweep watchdog. Two signals, deliberately from different sources:
#   - crash lines, streamed from the slurm logs (byte-offset tracked, so each
#     line is emitted once rather than re-emitted on every poll)
#   - progress + rollout health, read from the rollout dumps, NOT the logs:
#     stdout buffering makes the logs lag arbitrarily (arm 745395_1 sat at
#     "step 50" in the log while its dumps were at 84), so the dumps are the
#     only trustworthy liveness signal.
cd /home/ayf7/repos/abstention-reasoning || exit 1
JOBS="745395_0 745395_1 763768_1"
declare -A OFF DONE
for j in $JOBS; do OFF[$j]=$(stat -c %s "logs/hint_psweep_$j.out" 2>/dev/null || echo 0); done
tick=0
while true; do
  for j in $JOBS; do
    f=logs/hint_psweep_$j.out
    if [ -f "$f" ]; then
      new=$(stat -c %s "$f" 2>/dev/null || echo 0); old=${OFF[$j]:-0}
      if [ "$new" -gt "$old" ]; then
        tail -c +$((old+1)) "$f" 2>/dev/null | tr '\r' '\n' | grep -aE \
          "Traceback|out of memory|OutOfMemory|RuntimeError|AssertionError|srun: error|slurmstepd|Killed|CANCELLED|DUE TO (TIME|PREEMPTION|NODE)|Error executing job|died unexpectedly|Ray.*failed" \
          | head -4 | sed "s/^/[$j] /"
        OFF[$j]=$new
      fi
    fi
    if [ -z "${DONE[$j]}" ] && ! squeue -j "${j%%_*}" -h -o "%i" 2>/dev/null | grep -qx "$j"; then
      DONE[$j]=1
      echo "ARM $j LEFT THE QUEUE -- $(sacct -j "$j" -n -o State%20 2>/dev/null | head -1 | tr -d ' ')"
    fi
  done
  [ $((tick % 6)) -eq 0 ] && echo "heartbeat: $(python3 scripts/analysis/check_rollouts.py 2>&1 | tail -1)"
  tick=$((tick+1)); sleep 300
done
