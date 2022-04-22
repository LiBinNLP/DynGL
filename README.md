# DyGLSDP


Dynamic Graph Learning for Semantic Dependency Parsing. This repo is a extension of [Supar](https://github.com/yzhangcs/parser)


## Installation

`DyGLSDP` can be installed from source, you can also clone the repo and run it in IDE (i.e. Pycharm):
```sh
$ git clone https://github.com/LiBinNLP/DyGLSDP && cd parser
$ python setup.py install
```

As a prerequisite, the following requirements should be satisfied:
* `python`: >= 3.7
* [`pytorch`](https://github.com/pytorch/pytorch): >= 1.7
* [`transformers`](https://github.com/huggingface/transformers): >= 4.0

## Usage

### Parsing

You can download the pretrained model in BaiduNetDisk, and put them in the output directory:
```
(链接)URL：https://pan.baidu.com/s/1Gh1Fq-O_UbQaX0VKBHC5hA 
(提取码)Password：star
```

`DyGLSDP` allows you parse sentences with a few lines of code, example code is shown in [tests/test_sdp.py](https://github.com/LiBinNLP/DyGLSDP/blob/main/tests/test_sdp.py):
```py
>>> from supar import Parser 
>>> parser = Parser.load('/mnt/sda1_hd/atur/libin/projects/DyGLSDP/output/gcn/PSD/english/tag/model') 
>>> dataset = parser.predict('There is no asbestos in our products now .', lang='en', prob=True, verbose=False) 
>>> print(dataset[0])

```
By default, we use [`stanza`](https://github.com/stanfordnlp/stanza) internally to tokenize plain texts for parsing.
You only need to specify the language code `lang` for tokenization.

```py
>>> 
1	There	_	_	_	_	_	_	0:root	_
2	is	_	_	_	_	_	_	1:orphan	_
3	no	_	_	_	_	_	_	4:RSTR	_
4	asbestos	_	_	_	_	_	_	2:ACT-arg	_
5	in	_	_	_	_	_	_	1:orphan	_
6	our	_	_	_	_	_	_	7:APP	_
7	products	_	_	_	_	_	_	4:LOC	_
8	now	_	_	_	_	_	_	2:TWHEN	_
9	.	_	_	_	_	_	_	1:orphan	_

>>> 
```

### Training

To train a model of DyGLSDP, you need to specify some configurations in dygl-sdp.ini, and then use the following command line to start training:
```sh
$ python -m supar.cmds.dygl_sdp.py
train
-b
-d
0
-p
/mnt/sda1_hd/atur/libin/projects/DyGLSDP/output/gcn/DM/english/tag/model
-c
/mnt/sda1_hd/atur/libin/projects/DyGLSDP/dygl-sdp.ini
```


### Evaluation

To evaluate trained model, you can use the following command:
```sh
$ python -m supar.cmds.dygl_sdp.py
evaluate
-d
0
-p
/mnt/sda1_hd/atur/libin/projects/DyGLSDP/output/gcn/DM/english/tag/model
-c
/mnt/sda1_hd/atur/libin/projects/DyGLSDP/dygl-sdp.ini
```

## Performance

`DyGLSDP` provides pretrained models on SemEval-2015 Task 18 dataset for English, Chinese and Czech. 
Embedding settings:
Basic: Part-of-Speech tag embedding
+Char+Lemma: Part-of-Speech tag embedding + character embedding (CharLSTM) + lemma embedding
+Char+Lemma+BERT: Part-of-Speech tag embedding + character embedding (CharLSTM) + lemma embedding + BERT(base)

The tables below list the performance and parsing speed of pretrained models for different tasks.
All results are tested on the machine with Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz and Nvidia GeForce GTX 2080 Ti GPU.

### English
----------------------------------------------------------------------------------------
                               |      DM     |     PAS     |      PSD    |    Avg
                               |   ID | OOD  |   ID   OOD  |   ID    OOD |  ID  | OOD  |
----------------------------------------------------------------------------------------
DyGLSDP(GCN):Basic             | 93.7 | 89.3 | 94.9 | 91.7 | 85.9 | 84.1 | 91.5 | 88.5 |     
DyGLSDP(GAT):Basic             | 93.8 | 89.2 | 95.1 | 92.0 | 85.9 | 83.8 | 91.6 | 88.3 |
----------------------------------------------------------------------------------------
DyGLSDP(GCN):+Char+Lemma       | 95.0 | 90.1 | 95.0 | 92.0 | 86.6 | 85.0 | 92.2 | 89.0 |
DyGLSDP(GAT):+Char+Lemma       | 94.9 | 90.5 | 95.3 | 92.1 | 86.7 | 85.0 | 92.3 | 89.2 |
----------------------------------------------------------------------------------------
DyGLSDP(GCN):+Char+Lemma+BERT  | 95.8 | 92.7 | 96.2 | 94.2 | 87.8 | 87.0 | 93.3 | 91.3 | 
DyGLSDP(GAT):+Char+Lemma+BERT  | 95.9 | 92.7 | 96.2 | 94.3 | 87.7 | 87.2 | 93.3 | 91.4 |
----------------------------------------------------------------------------------------
### Chinese

### Czech
