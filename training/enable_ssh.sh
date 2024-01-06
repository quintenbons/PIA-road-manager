#!/bin/bash
echo "Preparing connections"
read -p "Enter your ensimag username: " username
echo ""
read -sp "Enter your ensimag password: " password
echo ""

TRAINING_DIR=$(dirname "$0")
source "$TRAINING_DIR/hosts_list.sh"
for host in "${HOSTS[@]}"; do
    echo "Adding $host to known hosts"
    ssh-keyscan -H $host 2> /dev/null >> ~/.ssh/known_hosts
    echo "Connecting to $host"
    sshpass -p $password ssh $username@$host "echo ok for host $host" || echo "failed host $host" &
done
wait
