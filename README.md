# CAL
Code for CAL
## For TUDataset
``dataset`` can be MUTAG NCI1 DD COLLAB IMDB-BINARY IMDB-MULTI

```
c=0.5
o=0.5
co=1.0
hidden=128
layer=4
cat_or_add=add
dataset=NCI1

CUDA_VISIBLE_DEVICES=$GPU python -u main_base.py \
--model $MODEL \
--dataset $dataset \
--c $c --o $o --co $co \
--hidden $hidden \
--layers $layer \
--seed 666 \
--cat_or_add $cat_or_add
```

## For SYN
```
GPU=$1
seed=666
model=CausalGIN

bs=128
bias=0.5
train_type=base

CUDA_VISIBLE_DEVICES=$GPU \
python -u main_toy.py --model $model --batch_size $bs --bias $bias --seed $seed --train_type $train_type --lr 0.001
```


