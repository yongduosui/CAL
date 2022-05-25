lr=0.002
min=1e-4
b=0.9
model=CausalGAT
CUDA_VISIBLE_DEVICES=$1 python -u main.py --bias $b --lr $lr --min_lr $min --model $model


