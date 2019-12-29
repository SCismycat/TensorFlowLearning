#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/24 18:22
# @Author  : Leslee

import argparse
parser = argparse.ArgumentParser(prog='demo',description='A demo program',
                                 epilog='The end of usage')

parser.add_argument('name')
parser.add_argument('-a','--age',type=int,required=True)
parser.add_argument('-s','--status',choices=['alpha','beta','released'],
                    type=str,dest='myStatus')
parser.print_help()


args = parser.parse_args()
args_known,args_unknown = parser.parse_known_args()
print(args)










