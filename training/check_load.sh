#!/bin/bash
TRAINING_DIR=$(dirname "$0")
source "$TRAINING_DIR/hosts_list.sh"
TMP_FILE="./tmp/loads.tmp"

read -sp "Enter your ensimag password: " password
echo ""

if [ ! -d ./tmp ]; then
    mkdir -p ./tmp
fi

> "$TMP_FILE"

fetch_load() {
    host=$1
    load=$(sshpass -p $password ssh -n $host "grep 'model name' /proc/cpuinfo | wc -l; cat /proc/loadavg" 2>/dev/null)

    read cores loadavg <<<$(echo "$load" | sed 'N;s/\n/ /')
    read one_min five_min fifteen_min rest <<< "$loadavg"

    if [[ -z "$one_min" ]]; then
        echo "No load data on $host. Cores: '$cores', 1-min Load Average: '$one_min'" >> "$TMP_FILE"
    else
        load_per_core=$(echo "$one_min" | bc -l)
        printf "%s: %.2f/%d\n" "$host" "$load_per_core" "$cores" >> "$TMP_FILE"
    fi
}

for host in "${HOSTS[@]}"; do
    fetch_load "$host" &
done

wait
cat "$TMP_FILE"
rm "$TMP_FILE"
