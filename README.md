# DyGLSDP


Dynamic Graph Learning for Semantic Dependency Parsing.


## Installation

`DyGLSDP` can be installed from source:
```sh
$ git clone https://github.com/LiBinNLP/DyGLSDP && cd parser
$ python setup.py install
```

As a prerequisite, the following requirements should be satisfied:
* `python`: >= 3.7
* [`pytorch`](https://github.com/pytorch/pytorch): >= 1.7
* [`transformers`](https://github.com/huggingface/transformers): >= 4.0

## Usage

`DyGLSDP` allows you to download the pretrained model and parse sentences with a few lines of code:
```py
>>> from supar import Parser
>>> parser = Parser.load('dygl-semantic-dependency')
>>> dataset = parser.predict('There is no asbestos in our products now .', lang='en', prob=True, verbose=False)
```
By default, we use [`stanza`](https://github.com/stanfordnlp/stanza) internally to tokenize plain texts for parsing.
You only need to specify the language code `lang` for tokenization.

The call to `parser.predict` will return an instance of `supar.utils.Dataset` containing the predicted results.
You can either access each sentence held in `dataset` or an individual field of all results.
Probabilities can be returned along with the results if `prob=True`.
```py
>>> 
1	There	there	EX	EX	_	0	root	0:root	_
2	is	is	VBZ	VBZ	_	1	orphan	1:orphan	_
3	no	no	DT	DT	_	1	orphan	1:orphan	_
4	asbestos	asbestos	NN	NN	_	2	ARG1	2:ARG1|3:BV|5:ARG1	_
5	in	in	IN	IN	_	1	orphan	1:orphan	_
6	our	our	PRP$	PRP$	_	1	orphan	1:orphan	_
7	products	product	NNS	NNS	_	5	ARG2	5:ARG2|6:poss	_
8	now	now	RB	RB	_	1	orphan	1:orphan	_
9	.	_	.	.	_	1	orphan	1:orphan	_
10	”	_	''	''	_	1	orphan	1:orphan	_

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

The evaluation process resembles prediction:
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

See [EXAMPLES](EXAMPLES.md) for more instructions on training and evaluation.

## Performance

`DyGLSDP` provides pretrained models on SemEval-2015 Task 18 dataset for English, Chinese and Czech. You can download them in BaiduNetDisk:
URL：https://pan.baidu.com/s/1Gh1Fq-O_UbQaX0VKBHC5hA 
Password：star

The tables below list the performance and parsing speed of pretrained models for different tasks.
All results are tested on the machine with Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz and Nvidia GeForce GTX 2080 Ti GPU.

