# Cross-lingual Phrase Retriever
This repository contains the code and pre-trained models for our paper XPR: Cross-lingual Phrase Retriever.

**************************** **Updates** ****************************

- 5/10 We released our [model checkpoint](./model), [evaluation code](./predict.py) and [our dataset](https://drive.google.com/file/d/1UEHqQhVj1V4mwKqs5iNdOxvRKurLUfoF/view?usp=sharing)
- 4/19 We released [our paper](https://arxiv.org/abs/2204.08887). Check it out!

## Overview


We propose a cross-lingual phrase retriever that extracts phrase representations from unlabeled example sentences named XPR.

![](figure/overview_xpr.png)

## Dataset

We also create a large-scale cross-lingual phrase retrieval dataset, which contains 65K bilingual phrase pairs and 4.2M example sentences in 8 English-centric language pairs.

## Requirements


Download XLMR checkpoint from Huggingface page: [link](https://huggingface.co/xlm-roberta-base).

Download XPR checkpoint from Huggingface:
[link](not finished)


```bash
pip install -r requirements.txt
mkdir model
mkdir log
```
<!-- 
## Directory Structure

```
XPR
├── DictMatching
│   ├── Loss.py                             XpCo loss
│   ├── moco.py                             MoCo/XPR model
│   └── simclr.py                           Simclr model
├── README.md
├── result
├── data                                    Dataset: Details in tree.txt
    ├── dev
    ├── merge_dataset.py
    ├── phrase
    ├── sentences
    ├── test
    └── train
├── inference.py                            Inference
├── predict.py                              Dev
├── requirements.txt
├── trainMoCo.py                            TrainXPR
└── utilsWord
    ├── args.py                             Args
    ├── sentence_process.py                 Add example sentences
    └── tools.py

```
 -->


## Run

<!-- Train our method:

```
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch  --nproc_per_node=2 \
--master_port 29501 trainMoCo.py --output_log_dir log_output --seed 42 \
--T_para 0.06 --simclr 0 --quene_length 0  --all_sentence_num 32 --train_sample_num 4 \
--dev_sample_num 32 --dev_only_q_encoder 1 --lg 'fr'
``` -->

Test our method:
- Download the mode file and move it into ./model/
- Make sure the path in test.sh is the path of dataset

```
bash test.sh
```
<!-- ```
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 predict.py  --lg 'fr' --sn '32' --test_lg 'fr' \
--output_log_dir 'test_result' --simclr 0 --quene_length 0 --T_para 0.06 --seed 42 --test_dev 0 --unsupervised 0 --wolinear 0
``` -->



## Results:

### Unsupervised Setting

|Model|ar-en|de-en|en-es|en-fr|en-ja|en-ko|en-ru|en-zh|avg|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|CLWE|2.74|0.78|0.00|1.02|0.34|0.28|1.32|0.12|0.83|
|CLSE|9.70|19.10|29.21|20.89|4.83|11.50|16.98|8.76|15.12|
|XPR|**14.71**|**28.96**|**42.25**|**39.38**|**7.34**|**15.22**|**24.24**|**11.26**|**22.92**|


### Supervised Setting

|Model|ar-en|de-en|en-es|en-fr|en-ja|en-ko|en-ru|en-zh|avg|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|CLWE|56.14|33.62|63.71|51.26|31.62|50.14|38.67|30.02|44.40|
|CLSE|20.58|18.79|36.06|26.60|16.73|24.58|21.32|17.69|22.79|
|XPR|**88.63**|**81.44**|**84.53**|**80.18**|**87.32**|**80.83**|**91.00**|**77.62**|**83.94**|

### Zero-shot  (ZH) Setting

|Model|ar-en|de-en|en-es|en-fr|en-ja|en-ko|en-ru|en-zh|avg|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|CLWE|0.04|0.32|0.22|0.23|0.00|2.24|0.09|30.02|4.15|
|CLSE|6.18|10.25|16.07|10.39|6.73|9.75|8.35|17.69|10.68|
|XPR|**74.12**|**73.60**|**82.54**|**77.36**|**73.04**|**78.52**|**79.10**|**77.62**|**76.99**|

### Multi-lingual supervised Setting

|Model|ar-en|de-en|en-es|en-fr|en-ja|en-ko|en-ru|en-zh|avg|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|CLWE|12.33|1.87|6.63|3.77|18.46|4.00|9.84|11.19|8.51|
|CLSE|11.98|19.64|29.44|21.58|11.91|14.73|18.01|14.50|17.72|
|XPR|**91.90**|**82.76**|**90.79**|**85.16**|**90.16**|**88.22**|**93.09**|**86.47**|**88.57**|
