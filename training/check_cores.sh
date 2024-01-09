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
    corenum=$(timeout 10 sshpass -p $password ssh -n $host "nproc" 2>/dev/null)

    if [[ -z "$corenum" ]]; then
        echo "No load data on $host. Cores: '$corenum'" >> "$TMP_FILE"
        return
    fi

    printf "%s: %d\n" "$host" "$corenum" >> "$TMP_FILE"
}

for i in $(seq 200 210); do
    formatted_number=$(printf "%03d" $i)
    host="ensipc${formatted_number}.ensimag.fr"
    fetch_load "$host" &
done

wait
cat "$TMP_FILE"
rm "$TMP_FILE"

