#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : sshuair
# @Time    : 2017/4/28

# *****************************************************
# generate multi label train file list
# *****************************************************

import os
import argparse
import pandas as pd
from sklearn.cross_validation import train_test_split


def oneVSall(inputfp, category):
    result = []
    # target = []
    with open(inputfp, 'r') as f:
        next(f)
        for idx, line in enumerate(f):
            # print(idx, line)
            line = line.rstrip()
            filename = line.split(',')[0]
            line_category = line.split(',')[1].split(' ')
            line_target = [0] * len(category)
            for item in line_category:
                index = targets.index(item)
                line_target[index] = 1
                # print(line_target)
            line_target.insert(0,idx)
            line_target.append(filename)
            result.append(line_target)
    return result


def phare_args():
    phare = argparse.ArgumentParser(description='phare the input text file to oneVSall formart')
    phare.add_argument('inputpath', help='input file')
    phare.add_argument('outoutpath', help='outpath file')
    args = phare.parse_args()
    # print(args.filepath)
    return args


def csv_to_lst():
    # args = phare_args()
    targets = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear', 'cloudy', 'conventional_mine',
               'cultivation', 'habitation', 'haze', 'partly_cloudy', 'primary', 'road', 'selective_logging', 'slash_burn', 'water']
    inputfp = '../data/train_v2.csv'
    output = '../data/train_v2.lst'
    result = oneVSall(inputfp, targets)
    with open(output,'w') as f:
        # f.write('index\t'+'\t'.join(targets)+'\tfile')
        for item in result:
            item = [str(x) for x in item]
            f.write('\t'.join(item) + '.jpg\n')


def prepare_test_lst():
    with open('../data/test_v2.lst','w') as f:
        for root, dir, files in os.walk('../data/test-jpg'):
            for idx, item in enumerate(files):
                print(item)
                f.writelines(str(idx) + '\t' + '0' + '\t' + item+'\n')


def split_train_val(val_num=8000):
    inputfp = '../data/train_v2.lst'
    df = pd.read_csv(inputfp, sep='\t', index_col=0, header=None)
    # df_train_sample = df.sample(n=val_num)
    # df_val_sample = df.sample(n=4000)

    train, val = train_test_split(df, test_size=8000)

    train.to_csv('../data/split/train_' + str(train.shape[0]) + '.lst', sep='\t', header = None)
    val.to_csv('../data/split/val_' + str(val.shape[0]) + '.lst', sep='\t',  header=None)

if __name__ == '__main__':
    pass
    # csv_to_lst()  # 原始文件转成0   1   0   0   1   1格式
    # prepare_test_lst()  # 生成比赛的test list
    split_train_val()  # 抽取部分数据作为sample