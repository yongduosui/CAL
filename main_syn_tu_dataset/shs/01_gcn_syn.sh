lr=0.002
min=5e-6
b=0.9
model=CausalGCN
CUDA_VISIBLE_DEVICES=$1 python -u main.py --bias $b --lr $lr --min_lr $min --model $model 




