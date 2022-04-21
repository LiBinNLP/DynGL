#!D:/Code/python
# -*- coding: utf-8 -*-
# @Time : 2022/4/21 23:05
# @Author : libin
# @File : test_sdp.py
# @Software: PyCharm

from supar import Parser
parser = Parser.load('/mnt/sda1_hd/atur/libin/projects/DyGLSDP/output/gcn/PSD/english/tag/model')
dataset = parser.predict('There is no asbestos in our products now .', lang='en', prob=True, verbose=False)
print(dataset[0])