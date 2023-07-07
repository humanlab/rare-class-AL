# Active Learning for Rare Class
## Details

This code accompanies the AL strategies used in our paper [Transfer and Active Learning for Dissonance Detection: Addressing the Rare Class Challenge](https://arxiv.org/abs/2305.02459). The five AL methods: 

* Random
* Entropy
* [CoreSet](https://arxiv.org/abs/1708.00489v4)
* [Contrastive Active Learning](https://aclanthology.org/2021.emnlp-main.51/)
* Probability-of-Rare-Class (PRC)

These are implemented separately from the dataset introduced in our paper so that they can be used out-of-the-box on any dataset.

![Active learning with needle in haystack](./images/something.png)

The PRC approach is a simple approach that picks examples most likely to be classified as the rare class by the model in each round of the active learning loop. This helps alleviate the needle-in-haystack problem which is prevalent in cases of absolute rarity (data scarcity combined with rare classes -- leading to almost no examples being found when data is collected from scratch.)
