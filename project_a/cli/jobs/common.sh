# Sourced by the job scripts: environment on the BU SCC (see docs/SCC.md).
# Override any of these by exporting it before qsub (qsub -V passes your environment)
# or by setting it in your ~/.bashrc.
TEAM_ROOT="${TEAM_ROOT:-/projectnb/ec523kb/projects/CowenZeollaZinTan}"
# Default: samc's venv (matches requirements.txt). A team-owned envs/dl_env is planned (docs/SCC.md).
ROADSEG_ENV="${ROADSEG_ENV:-$TEAM_ROOT/envs/samc/midway}"
if [ -d "$ROADSEG_ENV/conda-meta" ]; then                           # conda environment
  module load miniconda
  conda activate "$ROADSEG_ENV"
else                                                                # venv (python -m venv), e.g. envs/samc/midway
  module load "${ROADSEG_PYTHON_MODULE:-python3/3.12.4}"            # the Python the venv was built from
  source "$ROADSEG_ENV/bin/activate"
fi
cd "$SGE_O_WORKDIR"
export ROADSEG_DATA="${ROADSEG_DATA:-/projectnb/ec523kb/projects/RoadSeg/dataset}"
export TORCH_HOME="${TORCH_HOME:-$TEAM_ROOT/cache/torch}"        # DINOv2 downloaded once, shared
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
WORKERS=$(( NSLOTS > 2 ? NSLOTS - 2 : 0 ))
echo "job=$JOB_NAME user=$USER data=$ROADSEG_DATA python=$(which python) slots=$NSLOTS workers=$WORKERS gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
