The below commands to train and test the model should be done from `root` level:

Get requirements with `pip install -r requirements.txt`

Download data with `bash scripts/download_data.sh`

Prepare subsets for training with `python scripts/prepare_subset.py`

Train RotoGAT with `python train_gat.py`

Evaluate recall@K and loss curves for object and relation objectives with 
`python evaluate.py \
  --annotations data/annotations.json \
  --checkpoint checkpoint_epoch10.pt \
  --batch-size 1 \
  --N 8`

  
