GPU=$1
# dataset=MUTAG
# dataset=NCI1
# dataset=PROTEINS
# dataset=COLLAB
# dataset=IMDB-BINARY
# dataset=IMDB-MULTI
for model in CausalGCN GCN
do
for dataset in MUTAG NCI1 PROTEINS COLLAB IMDB-BINARY IMDB-MULTI
do
CUDA_VISIBLE_DEVICES=$GPU python -u main_real.py --model $model --dataset $dataset
done
done