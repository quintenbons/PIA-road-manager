#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: $0 <ensimag_username> <branch> <gen_size> <map_folder>"
    echo "    ensimag_username: your ensimag username (default: same as local)"
    echo "    branch: the branch to use (default: main)"
    echo "    gen_size: each core will generate this many entries (default: 10)"
    echo "    map_folder: src/maps/build/GUI/... (default: src/maps/build/GUI/Training-4/Uniform)"
    echo "    Example: $0 nomp cpp 1000"
fi

ENSIMAG_USER=${1:-$(whoami)}
BRANCH=${2:-main}
GEN_SIZE=${3:-10}
MAP_FOLDER=${4:-src/maps/build/GUI/Training-4/Uniform}

read -sp "Enter your ensimag password: " password
echo ""

TRAINING_DIR=$(dirname "$0")
source "$TRAINING_DIR/hosts_list.sh"
MASTER_NODE="${HOSTS[0]}"
echo "Master node: $ENSIMAG_USER@$MASTER_NODE"
echo "target map: $MAP_FOLDER"

# Check if tmp folder exists
echo "============== ENSIMAG: checking tmp folder"
sshpass -p $password ssh "$ENSIMAG_USER@$MASTER_NODE" "[ -d \$HOME/PIA-road-manager/tmp ]"
if [ $? -ne 0 ]; then
    echo "tmp folder does not exist, safe to create"
else
    # check if user wants to delete the folder
    read -p "tmp folder already exists, delete? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting tmp folder"
        sshpass -p $password ssh "$ENSIMAG_USER@$MASTER_NODE" "rm -rf \$HOME/PIA-road-manager/tmp"
    else
        echo "Aborting"
        exit 1
    fi
fi

# clone and update repo
echo "============== ENSIMAG: cloning"
sshpass -p $password ssh "$ENSIMAG_USER@$MASTER_NODE" <<EOF > /dev/null 2>&1
    cd
    if [ ! -d ~/PIA-road-manager ]; then
        echo "Clonage du dépôt sysd"
        git clone git@github.com:quintenbons/PIA-road-manager.git
    fi

    echo "Mise à jour du dépôt sysd"
    cd ~/PIA-road-manager
    git fetch
    git reset --hard "origin/$BRANCH"
    git clean -fd
    git checkout origin/$BRANCH
    git branch -D $BRANCH
    git checkout -b $BRANCH "origin/$BRANCH"

    # Do this manually to wake up the machines
    # ./training/wake_ensipcs.sh
EOF

if [ $? -ne 0 ]; then
    echo "============== ENSIMAG: failed to clone repo"
    exit 1
fi

# Generate datasets on all hosts
echo "============== ENSIMAG: verifying if local python3 version exists"
pyenv_path="venv-torch"
sshpass -p $password ssh "$ENSIMAG_USER@$MASTER_NODE" "[ -d \$HOME/$pyenv_path ]"
if [ $? -ne 0 ]; then
    echo "no local python3 version found, using system python3"
    py_interpreter="python3"
else
    echo "$pyenv_path exists, using it"
    py_interpreter="~/$pyenv_path/bin/python3"
fi

echo "============== ENSIMAG: generating datasets"
echo "Note: configure the hosts with authorized ssh keys and accept fingerprints once before running this script. enable_ssh.sh can help with this."
if [ ! -d ./logs ]; then
    mkdir -p ./logs
fi

salt=0

for host in "${HOSTS[@]}"; do
    salt=$((salt + 36))
    LOG_PATH="./logs/$host.log"
    echo "============== ENSIMAG: Launching dataset generation on $host. Logs in $LOG_PATH"
    sshpass -p $password ssh "$ENSIMAG_USER@$host" <<EOT > $LOG_PATH 2>&1 &
        mkdir -p ~/PIA-road-manager/tmp
        cd ~/PIA-road-manager
        $py_interpreter ./src/gen_parallel_dataset.py --dest ~/PIA-road-manager/tmp/\$(hostname) --map_folder $MAP_FOLDER --base-seed-salt $salt $GEN_SIZE
EOT
done

echo "============== ENSIMAG: Waiting for all hosts to finish"
time wait

echo "============== ENSIMAG: merging datasets"
time sshpass -p $password ssh -n "$ENSIMAG_USER@$MASTER_NODE" "cd ~/PIA-road-manager && $py_interpreter ./src/merge_datasets.py --dest ~/PIA-road-manager/tmp/dataset.pt \$(find ./tmp -type f -name '*.pt' | tr '\n' ' ')"

DEST_PATH="./ensimag.pt"
echo "============== ENSIMAG: pulling remote dataset ~/PIA-road-manager/tmp/dataset.pt to local $DEST_PATH"

if [ -f "$DEST_PATH" ]; then
    read -p "Local file $DEST_PATH already exists, overwrite? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting. Dataset is available on remote at ~/PIA-road-manager/tmp/dataset.pt"
        exit 1
    fi
fi

scp "$ENSIMAG_USER@$MASTER_NODE:~/PIA-road-manager/tmp/dataset.pt" "$DEST_PATH"
