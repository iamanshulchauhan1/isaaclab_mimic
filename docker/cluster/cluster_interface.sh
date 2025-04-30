#!/usr/bin/env bash

#==
# Configurations
#==


# Exit if error occurs
set -e

# Set tab-stops
tabs 4

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#==
# Functions
#==

# Print warnings in red
display_warning() {
    echo -e "\033[31mWARNING: $1\033[0m"
}

# Compare version numbers
version_gte() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n 1)" == "$2" ]
}

# Check Docker/Apptainer versions
check_docker_version() {
    if ! command -v docker &> /dev/null; then
        echo "[Error] Docker is not installed!" >&2; exit 1
    fi
    local dv=$(docker --version | awk '{print $3}')
    local av=$(apptainer --version | awk '{print $3}')
    if [ "$dv" = "24.0.7" ] && [ "$av" = "1.2.5" ] || \
       version_gte "$dv" "27.0.0" && version_gte "$av" "1.3.4"; then
        echo "[INFO] Docker $dv & Apptainer $av are compatible."
    else
        display_warning "Docker $dv & Apptainer $av are untested."
    fi
}

# Ensure Docker image exists locally
check_image_exists() {
    if ! docker image inspect "$1" &> /dev/null; then
        echo "[Error] Image '$1' not found!" >&2; exit 1
    fi
}

# Ensure Singularity image exists on remote
check_singularity_image_exists() {
    image_name="$1"
    if ! ssh $SSH_OPTIONS $CLUSTER_LOGIN "[ -f $CLUSTER_SIF_PATH/$image_name.tar ]"; then
        echo "[Error] '$image_name' not found on $CLUSTER_LOGIN" >&2; exit 1
    fi
}

# Submit the job (SLURM/PBS/CUSTOM)
submit_job() {
    echo "[INFO] Arguments passed to job script: $*"
    case $CLUSTER_JOB_SCHEDULER in
        SLURM) job_script_file=submit_job_slurm.sh ;;
        PBS )  job_script_file=submit_job_pbs.sh  ;;
        CUSTOM)
            echo "[INFO] Submitting in CUSTOM mode (detached)…"

            # build a timestamp and log‐file path
            ts=$(date +%Y%m%d_%H%M%S)
            LOGDIR="$CLUSTER_ISAACLAB_DIR/logs"
            LOGFILE="job_${ts}.log"

            # ensure remote log dir exists
            ssh $SSH_OPTIONS $CLUSTER_LOGIN "mkdir -p '$LOGDIR'"

            # fire off the run_singularity.sh under nohup on the remote,
            # with stdout+stderr → logs/job_TIMESTAMP.log
            ssh -f $SSH_OPTIONS $CLUSTER_LOGIN \
              "cd '$CLUSTER_ISAACLAB_DIR' && \
               nohup bash docker/cluster/run_singularity.sh \
                 '$CLUSTER_ISAACLAB_DIR' 'isaac-lab-$profile' $* \
               > '$LOGDIR/$LOGFILE' 2>&1"

            echo "[INFO] Job launched and detached; remote logs will be in $LOGDIR/$LOGFILE"
            return 0
            ;;
        *)
            echo "[Error] Unsupported scheduler: $CLUSTER_JOB_SCHEDULER" >&2
            exit 1
            ;;
    esac

    ssh $SSH_OPTIONS $CLUSTER_LOGIN \
        "cd $CLUSTER_ISAACLAB_DIR && bash docker/cluster/\$job_script_file '$CLUSTER_ISAACLAB_DIR' 'isaac-lab-$profile' $*"
}

#==
# Main
#==

help() {
    cat <<EOF

usage: $(basename "$0") [-h] <command> [<profile>] [<job_args>...]

Commands:
  push [profile]    Build & push container image (defaults to 'base')
  job  [profile]    Sync code & submit a job

EOF
}

# Parse flags
while getopts ":h" opt; do
    case $opt in
        h) help; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; help; exit 1 ;;
    esac
done
shift $((OPTIND -1))

# Need at least a command
if [ $# -lt 1 ]; then
    echo "Error: command required" >&2; help; exit 1
fi

command=$1; shift
profile="base"

case $command in

  push)
    [ $# -eq 1 ] && profile=$1
    echo "[INFO] Pushing container (profile=$profile)"
    if ! command -v apptainer &> /dev/null; then
        echo "[INFO] Apptainer not installed; skipping" && exit 0
    fi
    check_image_exists "isaac-lab-$profile:latest"
    check_docker_version
    source "$SCRIPT_DIR/.env.cluster"
    mkdir -p "$SCRIPT_DIR/exports"
    rm -rf "$SCRIPT_DIR/exports/isaac-lab-$profile"*
    cd "$SCRIPT_DIR/exports"
    APPTAINER_NOHTTPS=1 apptainer build --sandbox --fakeroot \
        "isaac-lab-$profile.sif" "docker-daemon://isaac-lab-$profile:latest"
    tar -cvf "isaac-lab-$profile.tar" "isaac-lab-$profile.sif"
    ssh $SSH_OPTIONS $CLUSTER_LOGIN "mkdir -p $CLUSTER_SIF_PATH"
    scp $SSH_OPTIONS \
        "$SCRIPT_DIR/exports/isaac-lab-$profile.tar" \
        "$CLUSTER_LOGIN:$CLUSTER_SIF_PATH/isaac-lab-$profile.tar"
    ;;

    job)
        if [ $# -ge 1 ] && [ -f "$SCRIPT_DIR/../.env.$1" ]; then
            profile=$1; shift
        fi
        job_args="$*"
        echo "[INFO] Executing job (profile=$profile)"
        source "$SCRIPT_DIR/.env.cluster"

        current_datetime=$(date +"%Y%m%d_%H%M%S")
        CLUSTER_ISAACLAB_DIR="${CLUSTER_ISAACLAB_DIR}_${current_datetime}"

        # 🧠 RECOMMENDED: set CLUSTER_LOGIN to use your ~/.ssh/config alias
        CLUSTER_LOGIN=gpu

        # ✅ Establish reusable SSH connection
        echo "[INFO] Establishing reusable SSH connection to compute node..."
        ssh -MNf $CLUSTER_LOGIN || echo "[WARN] SSH multiplexing failed or already established."

        check_singularity_image_exists "isaac-lab-$profile"
        ssh $SSH_OPTIONS $CLUSTER_LOGIN "mkdir -p $CLUSTER_ISAACLAB_DIR"

        echo "[INFO] Syncing IsaacLab code (including usd/)..."
        rsync -e "ssh $SSH_OPTIONS" -rh \
            --exclude="*.git*" \
            --filter='+ /usd/***' \
            --filter=':- .dockerignore' \
            "$SCRIPT_DIR/../.." \
            "$CLUSTER_LOGIN:$CLUSTER_ISAACLAB_DIR"

        echo "[INFO] Submitting job script..."
        submit_job $job_args
        ;;

  *)
    echo "Error: unknown command '$command'" >&2; help; exit 1
    ;;
esac
