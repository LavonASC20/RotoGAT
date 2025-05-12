set -e

DATA_DIR="./data"
ANN_DIR="$DATA_DIR/annotations"

mkdir -p "$ANN_DIR"

echo "Downloading objects_v1.json.zip…"
wget -q --show-progress \
    -O "$ANN_DIR/objects_v1.json.zip" \
    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects_v1.json.zip"

echo "Downloading relationships_v1.json.zip…"
wget -q --show-progress \
    -O "$ANN_DIR/relationships_v1.json.zip" \
    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships_v1.json.zip"

echo "Downloading image_data.json.zip…"
wget -q --show-progress \
    -O "$ANN_DIR/image_data.json.zip" \
    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"

echo "All three archives downloaded to $ANN_DIR"
