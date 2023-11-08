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

pip install --no-index torch torchvision numpy scipy tqdm

# moving dataset and code to $SLURM_TMPDIR
echo "moving datasets"
cp ~/projects/rrg-eugenium/DatasetsBelilovsky/imagenet_data/ILSVRC2012_img* $SLURM_TMPDIR
echo "moving code"
cp -r ~/scratch/proj/pytorch_examples/imagenet/* $SLURM_TMPDIR
cd $SLURM_TMPDIR

echo "extract training images"
mkdir imagenet/train && mv ILSVRC2012_img_train.tar imagenet/train/ && cd imagenet/train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
#
# At this stage imagenet/train will contain 1000 compressed .tar files, one for each category
#
# For each .tar file:
#   1. create directory with same name as .tar file
#   2. extract and copy contents of .tar file into directory
#   3. remove .tar file
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# Create validation directory; move .tar file; change directory; extract validation .tar; remove compressed file
echo "extract validation images"
cd ../..
mkdir imagenet/val && mv ILSVRC2012_img_val.tar imagenet/val/ && cd imagenet/val && tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
# get script from soumith and run; this script creates all class directories and moves images into corresponding directories
# wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
# no internet connection use local files
echo "move validation images to folders"
mv $SLURM_TMPDIR/valprep.sh $SLURM_TMPDIR/imagenet/val
bash valprep.sh
mkdir $SLURM_TMPDIR/output

cd $SLURM_TMPDIR
find imagenet/train/ -name "*.JPEG" | wc -l
find imagenet/val/ -name "*.JPEG" | wc -l