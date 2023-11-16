# Causal Attention for Interpretable and Generalizable Graph Classification
PyTorch implementation for Causal Attention for Interpretable and Generalizable Graph Classification

Yongduo Sui, Xiang Wang, Jiancan Wu, Min Lin, Xiangnan He, Tat-Seng Chua

In KDD 2022: https://dl.acm.org/doi/abs/10.1145/3534678.3539366



## Overview 

In this work, we take a causal look at the GNN modeling for graph classification. With our causal assumption, the shortcut feature serves as a confounder between the causal feature and prediction. It tricks the classifier to learn spurious correlations that facilitate the prediction in in-distribution (ID) test evaluation, while causing the performance drop in out-of-distribution (OOD) test data. To endow the classifier with better generalization, we propose the Causal Attention Learning (CAL) strategy, which discovers the causal patterns and mitigates the confounding effect of shortcuts. Specifically, we employ attention modules to estimate the causal and shortcut features of the input graph. We then parameterize the backdoor adjustment of causal theory — combine each causal feature with various shortcut features. It encourages the stable relationships between the causal estimation and the prediction, regardless of the changes in shortcut parts and distributions.


## Dependencies


Please setup the environment following Requirements in this [repository](https://github.com/chentingpc/gfn#requirements).
Typically, you might need to run the following commands:
```
pip install torch==1.4.0
pip install torch-scatter==1.1.0 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==0.4.4 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==1.4.5 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==1.1.0 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric==1.1.0
pip install torch-vision==0.5.0
```


## Experiments

### For Synthetic datasets
```
model=CausalGCN
lr=0.002
b=0.9
min=5e-6
python main_syn.py --bias $b --lr $lr --min_lr $min --model $model 
```
### For TU datasets

```python main_real.py --model CausalGAT --dataset MUTAG```

## Acknowledgements

The backbone implementation is reference to [https://github.com/chentingpc/gfn](https://github.com/chentingpc/gfn).

## Citation
If you use our codes or checkpoints, please cite our paper:
```
@inproceedings{sui2022causal,
  title={Causal Attention for Interpretable and Generalizable Graph Classification},
  author={Sui, Yongduo and Wang, Xiang and Wu, Jiancan and Lin, Min and He, Xiangnan and Chua, Tat-Seng},
  booktitle={KDD},
  year={2022}
}
```
