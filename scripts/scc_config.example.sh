# Copy to scc_config.local.sh on SCC and fill in your allocation/environment.
# Paths stay in project storage; no passwords or tokens belong in this file.
TEAM_ROOT=/projectnb/ec523kb/projects/CowenZeollaZinTan
SCC_PROJECT=""                         # Your actual qsub -P allocation
SCC_REPO="$TEAM_ROOT/code/$USER/deep_learning_bu_project"
SCC_ENV="$TEAM_ROOT/envs/$USER/midway"
SCC_PYTHON_MODULE=""                   # Set from `module avail python3`
SCC_PYTHON=python3                     # Must resolve to Python 3.12 after module load
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
DATA_ROOT=/projectnb/ec523kb/projects/RoadSeg/dataset
OUTPUT_ROOT="$TEAM_ROOT/runs/$USER"
export TORCH_HOME="$TEAM_ROOT/cache/torch"
ENCODER_WEIGHTS="$TEAM_ROOT/cache/resnet18-imagenet.safetensors"
SCC_CPUS=4
SCC_GPU_MEMORY=24G
SCC_GPU_CAPABILITY=7.0
SCC_GPU_TYPE=""                       # Optional course-approved type from qgpus
SCC_WALLTIME=04:00:00
SCC_BASELINE_UPDATES=200
SCC_REFINER_UPDATES=200
# Empty means preserve the original targets when resuming. Set to extend a run.
SCC_RESUME_BASELINE_UPDATES=""
SCC_RESUME_REFINER_UPDATES=""
# Set this to an existing run when submitting `resume`; leave empty otherwise.
SCC_RUN_DIR="${SCC_RUN_DIR:-}"
export DATA_ROOT ENCODER_WEIGHTS
