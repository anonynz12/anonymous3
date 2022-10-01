## CacheGNN: Enhancing Graph Neural Networks with Global Information Caching

This repository contains the PyTorch implementation for our CacheGNN. Further details about CacheGNN can be found in our paper.

## Requirements

- Python 3.6

- PyTorch 1.10.1
- Scikit-Learn 1.12
- Scipy 1.5.2
- PyTorch_Geometric 2.1.0

## Data download

Please first download the dataset and unzip it into `data` directory.

Google Drive Link: https://drive.google.com/file/d/1Js_RMDL82sU-kj2AIa9Lvn4lFWUaQb9K/view?usp=sharing

## Run the demo

```python
python train.py --cuda_id 0 --model [graphsage/gat/gcn/sgc/gcnii/gat2conv] --hidden_dim 64 --eta 1 --log_dir ./log/graphsage_dblp --k 3 --epochs 50 --dataset ppi --lr 1e-5
```