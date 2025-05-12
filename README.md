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

For moderately-intense machines, increase `batch-size` and `N`. Also can increase the image sample size for testing at the top of the file in `scripts/prepare_subset.py` via `SUBSET_SIZE`. 
