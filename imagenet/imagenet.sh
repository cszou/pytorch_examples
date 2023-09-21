#!/bin/bash
#SBATCH --account=rrg-eugenium           # Prof Eugene
#SBATCH --cpus-per-task=1                # Ask for 1 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=12:00:00                  # The job will run for 7 hours
#SBATCH -o /scratch/vs2410/slurm-%j.out  # Write the log in $SCRATCH

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/myvirenv
source $SLURM_TMPDIR/myvirenv/bin/activate

echo "moving datasets"
cp ~/projects/rrg-eugenium/DatasetsBelilovsky/imagenet_data/* $SLURM_TMPDIR
echo "moving code"
cp ~/scratch/proj/pytorch_examples/imagenet/* $SLURM_TMPDIR
cd $SLURM_TMPDIR

# mv valprep.sh imagenet/val
echo "extract images"
bash extract_ILSVRC.sh
mkdir $SLURM_TMPDIR/output
ls $SLURM_TMPDIR/output

pip install --no-index torch torchvision

# seed=$1
# tta_epochs=$2
# tta_data_till_n=$3
# cl_adapt=$4
# cl_adapt_rule=$5
# init_T_before_tta_as_S=$6
# comments=$7


# echo "seed: $seed"
# echo "tta_epochs: $tta_epochs"
# echo "tta_data_till_n: $tta_data_till_n"
# echo "cl_adapt: $cl_adapt"
# echo "cl_adapt_rule: $cl_adapt_rule"
# echo "init_T_before_tta_as_S: $init_T_before_tta_as_S"
# echo "comments: "$comments" "


# echo "save_path: $SLURM_TMPDIR/output"

python main.py -a alexnet --epochs 5 --lr 0.01

cp -r $SLURM_TMPDIR/output $SCRATCH
cp $SLURM_TMPDIR/checkpoint.pth.tar $SCRATCH
cp $SLURM_TMPDIR/model_best.pth.tar $SCRATCH