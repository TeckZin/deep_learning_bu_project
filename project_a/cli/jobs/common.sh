# Sourced by the job scripts: environment on the BU SCC (see docs/SCC.md).
# Override any of these by exporting it before qsub (qsub -V passes your environment)
# or by setting it in your ~/.bashrc.
TEAM_ROOT="${TEAM_ROOT:-/projectnb/ec523kb/projects/CowenZeollaZinTan}"
module load miniconda
conda activate "${ROADSEG_ENV:-$TEAM_ROOT/envs/dl_env}"          # shared env, by path
cd "$SGE_O_WORKDIR"
export ROADSEG_DATA="${ROADSEG_DATA:-/projectnb/ec523kb/projects/RoadSeg/dataset}"
export TORCH_HOME="${TORCH_HOME:-$TEAM_ROOT/cache/torch}"        # DINOv2 downloaded once, shared
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
WORKERS=$(( NSLOTS > 2 ? NSLOTS - 2 : 0 ))
echo "job=$JOB_NAME user=$USER data=$ROADSEG_DATA env=$CONDA_PREFIX slots=$NSLOTS workers=$WORKERS gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
