import os
import pandas as pd
import numpy as np
import torch
import argparse



import csv
'''划分为训练集，验证集，测试集'''
def huafen(df, train_num, valid_num, test_num):
    train_data = []
    train_data.append(df.columns.tolist())
    valid_data = []
    valid_data.append(df.columns.tolist())
    test_data = []
    test_data.append(df.columns.tolist())


    # 将数据随机打乱
    df = df.sample(frac=1)

    for i in range(df.shape[0]):
        if len(train_data) < train_num:
            train_data.append(df.iloc[i].tolist())
            
        elif len(valid_data) < valid_num:
            valid_data.append(df.iloc[i].tolist())
            
        elif len(test_data) < test_num:
            test_data.append(df.iloc[i].tolist())


    # 将list写入csv文件
    with open('./data/train.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in train_data:
            writer.writerow(row)
            
    with open('./data/valid.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in valid_data:
            writer.writerow(row)
    
    with open('./data/test.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in test_data:
            writer.writerow(row)

def leibie(data, label_num):
    label_nums = [0 for k in range(label_num)]
    for i in range(data.shape[0]):
        label_nums[int(data.iloc[i][0])]+=1
    print(label_nums)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='数据标准化')
    parser.add_argument('--path', type=str, required=True, help='输入要进行划分的 CSV 文件的路径')
    args = parser.parse_args()

    data_path = args.path
    df = pd.read_csv(data_path)
    labels_col = df.columns[0]
    label_num = 0
    label_set = []
    for i in range(df.shape[0]):
        #print(df['Diabetes_binary'][i])
        label_set.append(df[labels_col][i])
    label_num = len(set(label_set))
    label_num
    data_num = df.shape[0]
    huafen(df, 0.8*data_num, 0.1*data_num, 0.1*data_num)
    train_path = './data/train.csv'
    train_df = pd.read_csv(train_path)
    print("训练集一共"+str(train_df.shape[0])+"条")

    valid_path = './data/valid.csv'
    valid_df = pd.read_csv(valid_path)
    print("验证集一共"+str(valid_df.shape[0])+"条")

    test_path = './data/test.csv'
    test_df = pd.read_csv(test_path)
    print("测试集一共"+str(test_df.shape[0])+"条")

    
    print("训练集标签数量")
    leibie(train_df, label_num)
    print("验证集标签数量")
    leibie(valid_df, label_num)
    print("测试集标签数量")
    leibie(test_df, label_num)
