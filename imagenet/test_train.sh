#!/bin/bash
# SBATCH --account=rrg-eugenium           # Prof Eugene
# SBATCH --cpus-per-task=16                # Ask for 16 CPUs
# SBATCH --gres=gpu:1                     # Ask for 1 GPU
# SBATCH --mem=32G                        # Ask for 32 GB of RAM
# SBATCH --time=12:00:00                  # The job will run for 12 hours
# SBATCH -o /scratch/vs2410/slurm-%j.out  # Write the log in $SCRATCH

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/myvirenv
source $SLURM_TMPDIR/myvirenv/bin/activate

# moving dataset and code to $SLURM_TMPDIR
echo "moving datasets"
cp ~/projects/rrg-eugenium/DatasetsBelilovsky/imagenet_data/* $SLURM_TMPDIR
echo "moving code"
cp ~/scratch/proj/pytorch_examples/imagenet/* $SLURM_TMPDIR
cd $SLURM_TMPDIR

echo "extract images"
bash extract_ILSVRC.sh
mkdir $SLURM_TMPDIR/output
# ls $SLURM_TMPDIR/output

mkdir $SCRATCH/$1

pip install --no-index torch torchvision

outputfloder = $1

python main.py -a alexnet -j 16 --epochs 1 --lr 0.01

cp -r $SLURM_TMPDIR/output $SCRATCH
cp $SLURM_TMPDIR/checkpoint.pth.tar $SCRATCH/$1/
cp $SLURM_TMPDIR/model_best.pth.tar $SCRATCH/$1/