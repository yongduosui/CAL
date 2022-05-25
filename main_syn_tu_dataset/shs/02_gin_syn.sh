lr=0.001
min=1e-6
b=0.9
model=CausalGIN
CUDA_VISIBLE_DEVICES=$1 python -u main.py --bias $b --lr $lr --min_lr $min --model $model
