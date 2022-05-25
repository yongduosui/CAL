GPU=$1
dataset=MNIST
model=CausalGIN
layer=4
c=0.5
o=1.0
co=0.5

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
--config 'configs/superpixels_graph_classification_GIN_MNIST_100k.json' \
--dataset $dataset \
--model $model \
--c $c --o $o --co $co \
--layer $layer \
--eval_random False \
--random_or_avg random