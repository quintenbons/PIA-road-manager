#!/bin/bash
DUR=$1
if [ -z "$DUR" ]; then
  DUR=infinity
fi

TRAINING_DIR=$(dirname "$0")
source "$TRAINING_DIR/hosts_list.sh"

echo "Locking SSH for $DUR seconds"

read -sp "Enter your ensimag password: " password
echo ""

for host in "${HOSTS[@]}"; do
  echo locking $host
  sshpass -p $password ssh -n $host "sleep $DUR" &
done

wait
