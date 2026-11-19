# Team space on the BU SCC

`/projectnb/ec523kb/projects/CowenZeollaZinTan` (below: `$TEAM`). One shared place for the
environment, caches, checkpoints and logs; each member works in their own clone, so several
implementations can be tried in parallel. Code layout inside the repo: [CodeStructure.md](CodeStructure.md).

## Layout

```
$TEAM/
├── README.md                              short version of this page
├── code/<login>/deep_learning_bu_project  your OWN clone, your own branch
├── envs/dl_env                            ONE shared conda env (jobs activate it by path)
├── cache/torch                            TORCH_HOME: DINOv2 weights, downloaded once
├── data/                                  derived data only: fold manifests, stress copies
├── runs/                                  ALL training output, shared
│   ├── h0/<encoder>_<scheme>/fold<F>/     H0 heads + plan.json: trained once per fold, reused by all
│   ├── <login>-<run>/fold<F>/             refiner checkpoints + run.json (settings + git commit)
│   └── logs/                              job logs
├── checkpoints/                           promoted results only (team decision); never train into it
└── submissions/<date>_<run>/              exported zips + their --check output
```

- The dataset is **not** copied: it stays in `/projectnb/ec523kb/projects/RoadSeg/dataset`
  (`$ROADSEG_DATA`). `data/` holds fold manifests (symlinks, KB) and stress copies (~1-1.5 GB per fold).
- Fold assignments live in the repo (`project_a/cli/splits/folds.csv`), so every clone uses the same folds.

## How outputs reach the shared folders

In your clone, `project_a/cli/weights` and `project_a/cli/logs` are symlinks (both gitignored):
```bash
cd $TEAM/code/<login>/deep_learning_bu_project
ln -s $TEAM/runs      project_a/cli/weights
ln -s $TEAM/runs/logs project_a/cli/logs
```
Everything the scripts write under `weights/` and `logs/` lands in `runs/`, visible to everyone.

## Conventions

- **Run folders are per person.** `jobs/train.sh` names a run `<login>-<"run" in the experiment file>`
  (or `--run NAME`), so two members never write into the same folder. `train.py` also refuses a run
  folder whose `run.json` has different settings, or an existing run without `--resume`.
- **H0 heads are shared.** `jobs/cv.sh` trains `runs/h0/<encoder>_<scheme>/fold<F>` only if it does not
  exist yet; everyone else reuses it. A different encoder gets its own folder automatically.
- **Promote, don't train into** `checkpoints/` and `submissions/`: copy a result there once the team agrees.
- **Experiments are files** in the repo (`project_a/cli/experiments/*.json`), committed on your branch,
  so `run.json` (which records the git commit) is enough to reproduce any run.

## Permissions

Access is per person (POSIX ACLs), not through the `ec523kb` group (that is the whole class).
Members: `fdzeolla`, `samc`, `tant1` (+ Zin's login, to be added); TAs (`ec523kbta`) can read.

Only a file's owner can change its ACL, so **each member runs this once** over what they own. The
second line sets the *default* ACL, so anything created later by anyone is open to every member:
```bash
TEAM=/projectnb/ec523kb/projects/CowenZeollaZinTan
find $TEAM -user $USER -exec setfacl -m u:fdzeolla:rwX,u:samc:rwX,u:tant1:rwX {} +
find $TEAM -user $USER -type d -exec setfacl -m d:u:fdzeolla:rwx,d:u:samc:rwx,d:u:tant1:rwx {} +
```
Check a folder with `getfacl -p <folder>`: every member should appear in both `user:` and `default:user:` lines.

## First-time setup (per member)

```bash
ssh scc
TEAM=/projectnb/ec523kb/projects/CowenZeollaZinTan
mkdir -p $TEAM/code/$USER && cd $TEAM/code/$USER
git clone https://github.com/TeckZin/deep_learning_bu_project.git && cd deep_learning_bu_project
ln -s $TEAM/runs project_a/cli/weights && ln -s $TEAM/runs/logs project_a/cli/logs
module load miniconda && conda activate $TEAM/envs/dl_env
cd project_a/cli && python -m unittest discover -s tests          # 35 tests
```
Once for the team (whoever is first), from your clone's `project_a/cli`:
```bash
cd $TEAM/code/$USER/deep_learning_bu_project/project_a/cli
conda create -p $TEAM/envs/dl_env python=3.12 -y && conda activate $TEAM/envs/dl_env
pip install -r requirements.txt
TORCH_HOME=$TEAM/cache/torch python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)"
export ROADSEG_DATA=/projectnb/ec523kb/projects/RoadSeg/dataset
python splits.py make        # then commit project_a/cli/splits/folds.csv
```
Batch jobs set `ROADSEG_DATA`, `TORCH_HOME` and the env themselves (`jobs/common.sh`; override with
`TEAM_ROOT`, `ROADSEG_ENV`, `TORCH_HOME`). Training: README section 9 or `bash jobs/cv.sh <experiment> 0`.
