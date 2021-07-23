# Exploring the Efficacy of Automatically Generated Counterfactuals for Sentiment Analysis

Authors: Linyi Yang, Jiazheng Li, Pádraig Cunningham, Yue Zhang, Barry Smyth and Ruihai Dong

## Abstract

While state-of-the-art NLP models have been achieving the excellent performance of a wide range of tasks in recent years, important questions are being raised about their robustness and their underlying sensitivity to systematic biases that may exist in their training and test data. Such issues come to be manifest in performance problems when faced with out-of-distribution data in the field. One recent solution has been to use counterfactually augmented datasets in order to reduce any reliance on spurious patterns that may exist in the original data. Producing high-quality augmented data can be costly and time-consuming as it usually needs to involve human feedback and crowdsourcing efforts. In this work, we propose an alternative by describing and evaluating an approach to automatically generating counterfactual data for the purpose of data augmentation and explanation. A comprehensive evaluation on several different datasets and using a variety of state-of-the-art benchmarks demonstrate how our approach can achieve significant improvements in model performance when compared to models training on the original data and even when compared to models trained with the benefit of human-generated augmented data.

<p align="center">
<img src="./Introduction.png" width="550" >
</p>

## Quick Start

```bash
conda create --name cfsa python=3.7 
conda activate cfsa
bash run.sh
```
