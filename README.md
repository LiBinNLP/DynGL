# Dynamic Graph Learning for Semantic Dependency Parsing


Code for the paper **DynGL-SDP:Dynamic Graph Learning for Semantic Dependency Parsing [(COLING 2022)]([https://github.com/stanfordnlp/stanza](https://aclanthology.org/2022.coling-1.351)) **.


## Installation

`DynGL-SDP` can be installed from source, you can download it and run it in command line or IDE (i.e. Pycharm):
<!--$ git clone https://github.com/LiBinNLP/DynGL-SDP -->
```sh
$ cd /projects/DynGL-SDP/parser
$ python setup.py install
```

As a prerequisite, the following requirements should be satisfied:
* `python`: >= 3.6
* [`pytorch`](https://github.com/pytorch/pytorch): >= 1.7
* [`transformers`](https://github.com/huggingface/transformers): >= 4.0

## Usage

### Parsing

You can download the pretrained model in BaiduNetDisk, and put them in the output directory:
```
(链接)URL：https://pan.baidu.com/s/1Gh1Fq-O_UbQaX0VKBHC5hA 
(提取码)Password：star
```

`DynGL-SDP` allows you parse sentences with a few lines of code, example code is shown in tests/test_sdp.py<!--[tests/test_sdp.py](https://github.com/LiBinNLP/DynGL-SDP/blob/main/tests/test_sdp.py)-->:
```py
>>> from supar import Parser 
>>> parser = Parser.load('/projects/DynGL-SDP/output/gcn/PSD/english/tag/model') 
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
### Dataset
Experiments are conducted in [SemEval-2015 Task 18 dataset](https://catalog.ldc.upenn.edu/LDC2016T10). Trial data has been provided in our code.

### Training

To train a model of DynGL-SDP, you need to specify some configurations in dygl-sdp.ini, and then use the following command line to start training:
```sh
$ python -m supar.cmds.dygl_sdp.py
train
-b
-d
0
-p
/projects/DynGL-SDP/output/gcn/DM/english/tag/model
-c
/projects/DynGL-SDP/dygl-sdp.ini
```


### Evaluation

To evaluate trained model, you can use the following command:
```sh
$ python -m supar.cmds.dygl_sdp.py
evaluate
-d
0
-p
/projects/DynGL-SDP/output/gcn/DM/english/tag/model
-c
/projects/DynGL-SDP/dygl-sdp.ini
```

## Performance

`DynGL-SDP` achieves start-of-the-art performance on SemEval-2015 Task 18 dataset for English, Chinese and Czech. Trained models are provided in above.

Three embedding settings in the runtime:

```
Basic: Part-of-Speech tag embedding.
+Char+Lemma: Part-of-Speech tag embedding + character embedding (CharLSTM) + lemma embedding.
+Char+Lemma+BERT: Part-of-Speech tag embedding + character embedding (CharLSTM) + lemma embedding + BERT(base).
```

The tables below list the performance and parsing speed of pretrained models for different tasks.
All results are tested on the machine with Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz and Nvidia GeForce GTX 2080 Ti GPU.

### English
```
------------------------------------------------------------------------------------------
                                 |      DM     |     PAS     |      PSD    |    Avg      |
                                 |   ID | OOD  |   ID   OOD  |   ID    OOD |  ID  | OOD  |
------------------------------------------------------------------------------------------
DynGL-SDP(GCN):Basic             | 93.7 | 89.3 | 94.9 | 91.7 | 85.9 | 84.1 | 91.5 | 88.5 |     
DynGL-SDP(GAT):Basic             | 93.8 | 89.2 | 95.1 | 92.0 | 85.9 | 83.8 | 91.6 | 88.3 |
------------------------------------------------------------------------------------------
DynGL-SDP(GCN):+Char+Lemma       | 95.0 | 90.1 | 95.0 | 92.0 | 86.6 | 85.0 | 92.2 | 89.0 |
DynGL-SDP(GAT):+Char+Lemma       | 94.9 | 90.5 | 95.3 | 92.1 | 86.7 | 85.0 | 92.3 | 89.2 |
------------------------------------------------------------------------------------------
DynGL-SDP(GCN):+Char+Lemma+BERT  | 95.8 | 92.7 | 96.2 | 94.2 | 87.8 | 87.0 | 93.3 | 91.3 | 
DynGL-SDP(GAT):+Char+Lemma+BERT  | 95.9 | 92.7 | 96.2 | 94.3 | 87.7 | 87.2 | 93.3 | 91.4 |
------------------------------------------------------------------------------------------
```
### Chinese and Czech

```
--------------------------------------------------------------
                                 | PAS(Chinese)|  PSD(Czech) |   
                                 |      ID     |   ID   OOD  |
--------------------------------------------------------------
DynGL-SDP(GCN):Basic             |     88.8    | 89.3 | 94.9 |  
DynGL-SDP(GAT):Basic             |     88.9    | 89.2 | 95.1 |
--------------------------------------------------------------
DynGL-SDP(GCN):+Char+Lemma       |     88.5    | 90.1 | 95.0 |
DynGL-SDP(GAT):+Char+Lemma       |     88.3    | 90.5 | 95.3 |
--------------------------------------------------------------
DynGL-SDP(GCN):+Char+Lemma+BERT  |     90.8    | 92.7 | 96.2 |
DynGL-SDP(GAT):+Char+Lemma+BERT  |     90.8    | 92.7 | 96.2 |
--------------------------------------------------------------
```

### Citation
Please cite our paper if you are interested.
```
@inproceedings{li2022dyngl,
	title = "{DynGL-SDP: Dynamic Graph Learning for Semantic Dependency Parsing",
	author = {Li, Bin  and Gao, Miao  and Fan, Yunlong  and Sataer, Yikemaiti  and Gao, Zhiqiang  and Gui, Yaocheng},
	booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},
	month = {oct},
	year = {2022},
	address = {Gyeongju, Republic of Korea},
	url = {https://aclanthology.org/2022.coling-1.351},
	pages = {3994--4004}
}
```
