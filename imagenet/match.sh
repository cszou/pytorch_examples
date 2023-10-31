#!/bin/bash
# SBATCH --account=rrg-eugenium           # Prof Eugene
# SBATCH --cpus-per-task=16                # Ask for 16 CPUs
# SBATCH --gres=gpu:1                     # Ask for 1 GPU
# SBATCH --mem=32G                        # Ask for 32 GB of RAM
# SBATCH --time=2:00:00                  # The job will run for 12 hours
# SBATCH -o /scratch/vs2410/slurm-%j.out  # Write the log in $SCRATCH

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/myvirenv
source $SLURM_TMPDIR/myvirenv/bin/activate

# moving dataset and code to $SLURM_TMPDIR
echo "moving datasets"
cp ~/projects/rrg-eugenium/DatasetsBelilovsky/imagenet_data/ILSVRC2012_img_val.tar $SLURM_TMPDIR
echo "moving code"
cp ~/scratch/proj/pytorch_examples/imagenet/* $SLURM_TMPDIR
cd $SLURM_TMPDIR

echo "extract images"
# Create validation directory; move .tar file; change directory; extract validation .tar; remove compressed file
mkdir imagenet/val && mv ILSVRC2012_img_val.tar imagenet/val/ && cd imagenet/val && tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
# get script from soumith and run; this script creates all class directories and moves images into corresponding directories
# wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
# no internet connection use local files
mv $SLURM_TMPDIR/valprep.sh $SLURM_TMPDIR/imagenet/val
bash valprep.sh
mkdir $SLURM_TMPDIR/output
# ls $SLURM_TMPDIR/output

pip install --no-index torch torchvision numpy scipy tqdm

outputfloder = $1

mkdir $SCRATCH/$1

python matching.py

cp -r $SLURM_TMPDIR/output $SCRATCH
# cp $SLURM_TMPDIR/checkpoint.pth.tar $SCRATCH/$1/
# cp $SLURM_TMPDIR/model_best.pth.tar $SCRATCH/$1/