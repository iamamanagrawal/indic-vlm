## download necessary models
mkdir -p models
git-lfs install
[ -d models/gemma-3-1b-it ] || git clone https://huggingface.co/google/gemma-3-1b-it models/gemma-3-1b-it
[ -d models/siglip-base-patch16-256-multilingual ] || git clone https://huggingface.co/google/siglip-base-patch16-256-multilingual models/siglip-base-patch16-256-multilingual

## download dataset
mkdir -p pretrain_data
[ -d pretrain_data/moondream2-coyo-2M-captions ] || git clone https://huggingface.co/datasets/ljnlonoljpiljm/moondream2-coyo-2M-captions pretrain_data/moondream2-coyo-2M-captions
python -m src.process.ingest

## run pretraining
python -m src.pretrain