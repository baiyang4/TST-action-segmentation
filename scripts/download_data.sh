#!/bin/bash
# Download I3D features for action segmentation datasets
# Source: Zenodo (MS-TCN / ASFormer shared data)
# Total size: ~30GB
#
# Usage: bash scripts/download_data.sh
# Or in tmux: tmux send-keys -t tst_setup:data 'bash scripts/download_data.sh' Enter

set -e

DATA_DIR="/scratch/users/astar/ares/baiy2/project/TAS/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Downloading I3D features from Zenodo (~30GB) ==="
echo "This will take several hours on slow connections."
echo "Started at: $(date)"

# Download with resume support (-c) so it can be restarted if interrupted
wget -c -O data.zip 'https://zenodo.org/api/records/3625992/files/data.zip/content'

echo "Download complete at: $(date)"
echo "=== Extracting ==="

unzip -o data.zip
rm data.zip

echo "=== Done ==="
echo "Data extracted to: $DATA_DIR"
ls -la "$DATA_DIR"
