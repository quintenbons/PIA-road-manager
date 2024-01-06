#!/bin/bash
TRAINING_DIR=$(dirname "$0")

source "$TRAINING_DIR/hosts_list.sh"
# TO="ensipc-wake@ensimag.fr"
TO="quinten.bons@grenoble-inp.org"
SUBJECT="Hello"

gen_body() {
  HOSTS=$1
  for host in "${HOSTS[@]}"; do
      echo "$host"
  done
}

echo "Sending mail for hosts ${HOSTS[@]}"
gen_body $HOSTS | mail -s "$SUBJECT" "$TO"
