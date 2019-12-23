#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/23 11:02
# @Author  : Leslee

"""
背景：用户自定义的二进制文件，存储对象是str。
1. 使用tf.FixedLengthRecordReader读取二进制文件中固定长度的字节块
2. 使用tf.decode_raw方法将字符串转换为uint8的张量。
3. 按照用户定义的数据结构将这些张量组织为输入样例
"""










