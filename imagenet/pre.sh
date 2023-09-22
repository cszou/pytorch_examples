module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/myvirenv
source $SLURM_TMPDIR/myvirenv/bin/activate
pip install --no-index torch torchvision