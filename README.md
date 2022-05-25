# Causal Attention for Interpretable and Generalizable Graph Classification
PyTorch implementation for Causal Attention for Interpretable and Generalizable Graph Classification

YongduoSui, Xiang Wang, Jiancan Wu, Min Lin, Xiangnan He, Tat-Seng Chua

In KDD 2022.



## Overview 

In this work, we take a causal look at the GNN modeling for graph classification. With our causal assumption, the shortcut feature serves as a confounder between the causal feature and prediction. It tricks the classifier to learn spurious correlations that facilitate the prediction in in-distribution (ID) test evaluation, while causing the performance drop in out-of-distribution (OOD) test data. To endow the classifier with better generalization, we propose the Causal Attention Learning (CAL) strategy, which discovers the causal patterns and mitigates the confounding effect of shortcuts. Specifically, we employ attention modules to estimate the causal and shortcut features of the input graph. We then parameterize the backdoor adjustment of causal theory — combine each causal feature with various shortcut features. It encourages the stable relationships between the causal estimation and the prediction, regardless of the changes in shortcut parts and distributions. Extensive experiments on synthetic and real-world datasets demonstrate the effectiveness of CAL.



## Experiments

- For Synthetic  datasets and TU datasets, please refer to ```main_syn_tu_dataset```
- For Superpixel datasets, please refer to ```main_superpixel```

