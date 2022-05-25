# Superpixel datasets experiments
## 1. Requirements
Please follow this [repo](https://github.com/graphdeeplearning/benchmarking-gnns) to create your environment and download datasets.

## 2. Command:
For GCN + CAL on MNIST dataset

```
python main.py \
--config 'configs/superpixels_graph_classification_GCN_MNIST_100k.json' \
--dataset MNIST \
--model CausalGCN \
--c 0.3 \
--o 1.0 \
--co 0.5 \
--layer 4 \
--eval_random False \
--random_or_avg random
```



