GPU=$1
dataset=MUTAG
model=CausalGAT
# dataset=NCI1
# dataset=PROTEINS
# dataset=COLLAB
# dataset=IMDB-BINARY
# dataset=IMDB-MULTI
# for model in CausalGIN GIN
# do
# for dataset in MUTAG NCI1 PROTEINS COLLAB IMDB-BINARY IMDB-MULTI
# do
# CUDA_VISIBLE_DEVICES=$GPU python -u main_real.py --model $model --dataset $dataset
# done
# done

CUDA_VISIBLE_DEVICES=$GPU python -u main_real.py --model $model --dataset $dataset