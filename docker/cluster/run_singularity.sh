#!/usr/bin/env bash

echo "(run_singularity.py): Called on compute node from current isaaclab directory $1 with container profile $2 and arguments ${@:3}"

#==
# Helper functions
#==
setup_directories() {
    # Check and create directories
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}

#==
# Main
#==

# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load variables to set the Isaac Lab path on the cluster
source $SCRIPT_DIR/.env.cluster
source $SCRIPT_DIR/../.env.base

# initialize environment modules if available
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
fi

# load apptainer/singularity modules if provided by the cluster
if command -v module &> /dev/null; then
    module load apptainer || true
    module load singularity || true
fi

# debug: print PATH and locate binaries
echo "[DEBUG] PATH=$PATH"
echo "[DEBUG] which apptainer: $(command -v apptainer || echo 'not found')"
echo "[DEBUG] which singularity: $(command -v singularity || echo 'not found')"

# Ensure TMPDIR exists (for non-scheduler environments)
# if [ -z "$TMPDIR" ]; then
#     TMPDIR=$(mktemp -d)
#     echo "[INFO] TMPDIR was empty, using $TMPDIR"
# fi

# Make sure your pre‑exported TMPDIR actually exists and log it
mkdir -p "$TMPDIR"
echo "[INFO] Using TMPDIR=$TMPDIR"



# Detect container runtime, allowing override via CONTAINER_RUNTIME
if [ -n "$CONTAINER_RUNTIME" ] && command -v "$CONTAINER_RUNTIME" &> /dev/null; then
    SINGULARITY_BIN="$CONTAINER_RUNTIME"
elif [ -x "/usr/bin/apptainer" ]; then
    SINGULARITY_BIN="/usr/bin/apptainer"
elif [ -x "/usr/local/bin/apptainer" ]; then
    SINGULARITY_BIN="/usr/local/bin/apptainer"
elif command -v apptainer &> /dev/null; then
    SINGULARITY_BIN=apptainer
elif [ -x "/usr/bin/singularity" ]; then
    SINGULARITY_BIN="/usr/bin/singularity"
elif command -v singularity &> /dev/null; then
    SINGULARITY_BIN=singularity
else
    echo "[ERROR] No container runtime found. Tried CONTAINER_RUNTIME, absolute paths, apptainer, and singularity." >&2
    exit 1
fi

# make sure that all directories exist in cache directory
setup_directories

# copy all cache files into a subfolder for rsync
mkdir -p "$TMPDIR/docker-isaac-sim"
cp -r "$CLUSTER_ISAAC_SIM_CACHE_DIR"/. "$TMPDIR/docker-isaac-sim/"

# make sure logs directory exists (in the permanent isaaclab directory)
mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
touch "$CLUSTER_ISAACLAB_DIR/logs/.keep"

# copy the temporary isaaclab directory with the latest changes to the compute node
cp -r "$1" "$TMPDIR"
# Get the directory name
dir_name=$(basename "$1")

# unpack the SIF tarball into TMPDIR
tar -xf "$CLUSTER_SIF_PATH/$2.tar" -C "$TMPDIR"

# make sure logs directory exists (in the permanent isaaclab directory)
mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
touch   "$CLUSTER_ISAACLAB_DIR/logs/.keep"

# make sure outputs directory exists (for Hydra / experiment outputs)
mkdir -p "$CLUSTER_ISAACLAB_DIR/outputs"
touch   "$CLUSTER_ISAACLAB_DIR/outputs/.keep"


# execute command in container
# NOTE: ISAACLAB_PATH is normally set in isaaclab.sh but we directly call the isaac-sim python
"$SINGULARITY_BIN" exec \
    -B "$TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache":rw \
    -B "$TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov":rw \
    -B "$TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip":rw \
    -B "$TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache":rw \
    -B "$TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache":rw \
    -B "$TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs":rw \
    -B "$TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data":rw \
    -B "$TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents":rw \
    -B "$TMPDIR/$dir_name:/workspace/isaaclab":rw \
    -B "$CLUSTER_ISAACLAB_DIR/logs:/workspace/isaaclab/logs":rw \
    -B "$CLUSTER_ISAACLAB_DIR/outputs:/workspace/isaaclab/outputs":rw \
    -B "$CLUSTER_ISAAC_SIM_CACHE_DIR/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw" \
    --nv --writable --containall "$TMPDIR/$2.sif" \
    bash -c 'export ISAACLAB_PATH=/workspace/isaaclab && \
             cd /workspace/isaaclab && \
             exec /isaac-sim/python.sh "$@"' _ "${CLUSTER_PYTHON_EXECUTABLE}" "${@:3}"

# copy resulting cache files back to host
rsync -azPv "$TMPDIR/docker-isaac-sim" "$CLUSTER_ISAAC_SIM_CACHE_DIR/.."

echo "(run_singularity.py): Return"
