#!/bin/bash
#SBATCH --ntasks=1               # total number of tasks across all nodes
##SBATCH --mem=16G
#SBATCH --time=30-00:00:00
##SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=a100_3g.40gb:1
##SBATCH --gpus-per-node=a100_7g.80gb:1

module purge
module load python/anaconda3

eval "$(conda shell.bash hook)"  
conda activate redenv 


python main.py \
	--dset=cifar100 \
	--dset_path=~/Datasets/cifar100 \
	--arch=wideresnet \
	--config_path=./configs/neurips/iht_cifar100_wideresnet_steplr_freq20_s50.yaml \
	--workers=8 \
	--epochs=200 \
	--warmup_epochs=5 \
	--batch_size=16 \
	--reset_momentum_after_recycling \
	--checkpoint_freq 50 \
        --manual_seed="12345" \
	--experiment_root_path "./experiments_iht" \
	--exp_name=cifar100_wideresnet 


## --gpu!!!!!!/bin/bash
#SBATCH --job-name=myjob         # create a short name for your job
#SBATCH --nodes=1             # node count
#SB