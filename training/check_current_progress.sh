#!/bin/bash
TRAINING_DIR=$(dirname "$0")
source "$TRAINING_DIR/hosts_list.sh"
LOG_FOLDER="./logs"

if [ ! -d ./logs ]; then
  echo "No ongoing generation"
  exit 0
fi

for host in "${HOSTS[@]}"; do
  LOG_PATH="$LOG_FOLDER/$host.log"
  if [ -f "$LOG_PATH" ]; then
    echo "============== ENSIMAG: $host"
    cat "$LOG_PATH" | grep "..%"
  fi
done
