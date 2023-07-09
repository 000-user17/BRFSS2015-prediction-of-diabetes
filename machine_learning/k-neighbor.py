import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    train_df = pd.read_csv('../data/train.csv')
    valid_df = pd.read_csv('../data/valid.csv')
    test_df = pd.read_csv('../data/test.csv')

    train_data = train_df.to_numpy()
    valid_data = valid_df.to_numpy()
    test_data = test_df.to_numpy()

    '''获取标签'''
    train_y = train_data[:,0]
    valid_y = valid_data[:,0]
    test_y = test_data[:,0]

    '''获取特征'''
    train_x = train_data[:, 1:]
    valid_x = valid_data[:, 1:]
    test_x = test_data[:, 1:]
    
    '''1：1的数据集'''
    new_df = pd.read_csv('../data/train.csv')
    new_data = new_df.to_numpy()
    new_data_x = new_data[:, 1:]
    new_data_y = new_data[:, 0]
    


    # 创建K近邻模型实例
    print("***********************KNeighborsClassifier(n_neighbors=5, weights='distance')***************************")
    clf = KNeighborsClassifier(n_neighbors=5, weights="distance")

    # 训练模型
    clf = clf.fit(train_x, train_y)

    # 使用模型进行预测
    pre_train = clf.predict(train_x)
    pre_valid = clf.predict(valid_x)
    pre_test = clf.predict(test_x)
    pre_new = clf.predict(new_data_x)

    # 计算准确率
    accuracy = sum(pre_train == train_y) / len(train_y)
    print("train Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_valid == valid_y) / len(valid_y)
    print("valid Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_test == test_y) / len(test_y)
    print("test Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_new == new_data_y) / len(new_data_y)
    print("new data Accuracy: {:.2f}%".format(accuracy*100))

    # 计算精确率
    precision = precision_score(train_y, pre_train)
    print("train Precision: {:.2f}%".format(precision*100))

    precision = precision_score(valid_y, pre_valid)
    print("valid Precision: {:.2f}%".format(precision*100))

    precision = precision_score(test_y, pre_test)
    print("test Precision: {:.2f}%".format(precision*100))

    precision = precision_score(new_data_y, pre_new)
    print("new data Precision: {:.2f}%".format(precision*100))

    # 计算召回率
    recall = recall_score(train_y, pre_train)
    print("train Recall: {:.2f}%".format(recall*100))

    recall = recall_score(valid_y, pre_valid)
    print("valid Recall: {:.2f}%".format(recall*100))

    recall = recall_score(test_y, pre_test)
    print("test Recall: {:.2f}%".format(recall*100))

    recall = recall_score(new_data_y, pre_new)
    print("new data Recall: {:.2f}%".format(recall*100))

    # 计算F1-score
    f1 = f1_score(train_y, pre_train)
    print("train F1-score: {:.2f}".format(f1))

    f1 = f1_score(valid_y, pre_valid)
    print("valid F1-score: {:.2f}".format(f1))

    f1 = f1_score(test_y, pre_test)
    print("test F1-score: {:.2f}".format(f1))

    f1 = f1_score(new_data_y, pre_new)
    print("new data F1-score: {:.2f}".format(f1))

    # 计算AUC-ROC
    auc = roc_auc_score(train_y, pre_train)
    print("train AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(valid_y, pre_valid)
    print("valid AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(test_y, pre_test)
    print("test AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(new_data_y, pre_new)
    print("new data AUC-ROC: {:.2f}".format(auc))

    # 计算混淆矩阵
    confusion_train = confusion_matrix(train_y, pre_train)
    print("train Confusion Matrix:")
    print(confusion_train)

    confusion_valid = confusion_matrix(valid_y, pre_valid)
    print("valid Confusion Matrix:")
    print(confusion_valid)

    confusion_test = confusion_matrix(test_y, pre_test)
    print("test Confusion Matrix:")
    print(confusion_test)

    confusion_new = confusion_matrix(new_data_y, pre_new)
    print("new data Confusion Matrix:")
    print(confusion_new)


    # 创建K近邻模型实例
    print("***********************KNeighborsClassifier(n_neighbors=5, weights='uniform')***************************")

    clf = KNeighborsClassifier(n_neighbors=5, weights="uniform")

    # 训练模型
    clf = clf.fit(train_x, train_y)

    # 使用模型进行预测
    pre_train = clf.predict(train_x)
    pre_valid = clf.predict(valid_x)
    pre_test = clf.predict(test_x)
    pre_new = clf.predict(new_data_x)

    # 计算准确率
    accuracy = sum(pre_train == train_y) / len(train_y)
    print("train Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_valid == valid_y) / len(valid_y)
    print("valid Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_test == test_y) / len(test_y)
    print("test Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_new == new_data_y) / len(new_data_y)
    print("new data Accuracy: {:.2f}%".format(accuracy*100))

    # 计算精确率
    precision = precision_score(train_y, pre_train)
    print("train Precision: {:.2f}%".format(precision*100))

    precision = precision_score(valid_y, pre_valid)
    print("valid Precision: {:.2f}%".format(precision*100))

    precision = precision_score(test_y, pre_test)
    print("test Precision: {:.2f}%".format(precision*100))

    precision = precision_score(new_data_y, pre_new)
    print("new data Precision: {:.2f}%".format(precision*100))

    # 计算召回率
    recall = recall_score(train_y, pre_train)
    print("train Recall: {:.2f}%".format(recall*100))

    recall = recall_score(valid_y, pre_valid)
    print("valid Recall: {:.2f}%".format(recall*100))

    recall = recall_score(test_y, pre_test)
    print("test Recall: {:.2f}%".format(recall*100))

    recall = recall_score(new_data_y, pre_new)
    print("new data Recall: {:.2f}%".format(recall*100))

    # 计算F1-score
    f1 = f1_score(train_y, pre_train)
    print("train F1-score: {:.2f}".format(f1))

    f1 = f1_score(valid_y, pre_valid)
    print("valid F1-score: {:.2f}".format(f1))

    f1 = f1_score(test_y, pre_test)
    print("test F1-score: {:.2f}".format(f1))

    f1 = f1_score(new_data_y, pre_new)
    print("new data F1-score: {:.2f}".format(f1))

    # 计算AUC-ROC
    auc = roc_auc_score(train_y, pre_train)
    print("train AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(valid_y, pre_valid)
    print("valid AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(test_y, pre_test)
    print("test AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(new_data_y, pre_new)
    print("new data AUC-ROC: {:.2f}".format(auc))

    # 计算混淆矩阵
    confusion_train = confusion_matrix(train_y, pre_train)
    print("train Confusion Matrix:")
    print(confusion_train)

    confusion_valid = confusion_matrix(valid_y, pre_valid)
    print("valid Confusion Matrix:")
    print(confusion_valid)

    confusion_test = confusion_matrix(test_y, pre_test)
    print("test Confusion Matrix:")
    print(confusion_test)

    confusion_new = confusion_matrix(new_data_y, pre_new)
    print("new data Confusion Matrix:")
    print(confusion_new)


    # 创建K近邻模型实例
    print("***********************KNeighborsClassifier(n_neighbors=4, weights='distance')***************************")

    clf = KNeighborsClassifier(n_neighbors=4, weights="distance")

    # 训练模型
    clf = clf.fit(train_x, train_y)

    # 使用模型进行预测
    pre_train = clf.predict(train_x)
    pre_valid = clf.predict(valid_x)
    pre_test = clf.predict(test_x)
    pre_new = clf.predict(new_data_x)

    # 计算准确率
    accuracy = sum(pre_train == train_y) / len(train_y)
    print("train Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_valid == valid_y) / len(valid_y)
    print("valid Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_test == test_y) / len(test_y)
    print("test Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_new == new_data_y) / len(new_data_y)
    print("new data Accuracy: {:.2f}%".format(accuracy*100))

    # 计算精确率
    precision = precision_score(train_y, pre_train)
    print("train Precision: {:.2f}%".format(precision*100))

    precision = precision_score(valid_y, pre_valid)
    print("valid Precision: {:.2f}%".format(precision*100))

    precision = precision_score(test_y, pre_test)
    print("test Precision: {:.2f}%".format(precision*100))

    precision = precision_score(new_data_y, pre_new)
    print("new data Precision: {:.2f}%".format(precision*100))

    # 计算召回率
    recall = recall_score(train_y, pre_train)
    print("train Recall: {:.2f}%".format(recall*100))

    recall = recall_score(valid_y, pre_valid)
    print("valid Recall: {:.2f}%".format(recall*100))

    recall = recall_score(test_y, pre_test)
    print("test Recall: {:.2f}%".format(recall*100))

    recall = recall_score(new_data_y, pre_new)
    print("new data Recall: {:.2f}%".format(recall*100))

    # 计算F1-score
    f1 = f1_score(train_y, pre_train)
    print("train F1-score: {:.2f}".format(f1))

    f1 = f1_score(valid_y, pre_valid)
    print("valid F1-score: {:.2f}".format(f1))

    f1 = f1_score(test_y, pre_test)
    print("test F1-score: {:.2f}".format(f1))

    f1 = f1_score(new_data_y, pre_new)
    print("new data F1-score: {:.2f}".format(f1))

    # 计算AUC-ROC
    auc = roc_auc_score(train_y, pre_train)
    print("train AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(valid_y, pre_valid)
    print("valid AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(test_y, pre_test)
    print("test AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(new_data_y, pre_new)
    print("new data AUC-ROC: {:.2f}".format(auc))

    # 计算混淆矩阵
    confusion_train = confusion_matrix(train_y, pre_train)
    print("train Confusion Matrix:")
    print(confusion_train)

    confusion_valid = confusion_matrix(valid_y, pre_valid)
    print("valid Confusion Matrix:")
    print(confusion_valid)

    confusion_test = confusion_matrix(test_y, pre_test)
    print("test Confusion Matrix:")
    print(confusion_test)

    confusion_new = confusion_matrix(new_data_y, pre_new)
    print("new data Confusion Matrix:")
    print(confusion_new)


    # 创建K近邻模型实例
    print("***********************KNeighborsClassifier(n_neighbors=4, weights='uniform')***************************")

    clf = KNeighborsClassifier(n_neighbors=4, weights="uniform")

    # 训练模型
    clf = clf.fit(train_x, train_y)

    # 使用模型进行预测
    pre_train = clf.predict(train_x)
    pre_valid = clf.predict(valid_x)
    pre_test = clf.predict(test_x)
    pre_new = clf.predict(new_data_x)

    # 计算准确率
    accuracy = sum(pre_train == train_y) / len(train_y)
    print("train Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_valid == valid_y) / len(valid_y)
    print("valid Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_test == test_y) / len(test_y)
    print("test Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_new == new_data_y) / len(new_data_y)
    print("new data Accuracy: {:.2f}%".format(accuracy*100))

    # 计算精确率
    precision = precision_score(train_y, pre_train)
    print("train Precision: {:.2f}%".format(precision*100))

    precision = precision_score(valid_y, pre_valid)
    print("valid Precision: {:.2f}%".format(precision*100))

    precision = precision_score(test_y, pre_test)
    print("test Precision: {:.2f}%".format(precision*100))

    precision = precision_score(new_data_y, pre_new)
    print("new data Precision: {:.2f}%".format(precision*100))

    # 计算召回率
    recall = recall_score(train_y, pre_train)
    print("train Recall: {:.2f}%".format(recall*100))

    recall = recall_score(valid_y, pre_valid)
    print("valid Recall: {:.2f}%".format(recall*100))

    recall = recall_score(test_y, pre_test)
    print("test Recall: {:.2f}%".format(recall*100))

    recall = recall_score(new_data_y, pre_new)
    print("new data Recall: {:.2f}%".format(recall*100))

    # 计算F1-score
    f1 = f1_score(train_y, pre_train)
    print("train F1-score: {:.2f}".format(f1))

    f1 = f1_score(valid_y, pre_valid)
    print("valid F1-score: {:.2f}".format(f1))

    f1 = f1_score(test_y, pre_test)
    print("test F1-score: {:.2f}".format(f1))

    f1 = f1_score(new_data_y, pre_new)
    print("new data F1-score: {:.2f}".format(f1))

    # 计算AUC-ROC
    auc = roc_auc_score(train_y, pre_train)
    print("train AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(valid_y, pre_valid)
    print("valid AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(test_y, pre_test)
    print("test AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(new_data_y, pre_new)
    print("new data AUC-ROC: {:.2f}".format(auc))

    # 计算混淆矩阵
    confusion_train = confusion_matrix(train_y, pre_train)
    print("train Confusion Matrix:")
    print(confusion_train)

    confusion_valid = confusion_matrix(valid_y, pre_valid)
    print("valid Confusion Matrix:")
    print(confusion_valid)

    confusion_test = confusion_matrix(test_y, pre_test)
    print("test Confusion Matrix:")
    print(confusion_test)

    confusion_new = confusion_matrix(new_data_y, pre_new)
    print("new data Confusion Matrix:")
    print(confusion_new)


    # 创建K近邻模型实例
    print("***********************KNeighborsClassifier(n_neighbors=3, weights='distance')***************************")

    clf = KNeighborsClassifier(n_neighbors=3, weights="distance")

    # 训练模型
    clf = clf.fit(train_x, train_y)

    # 使用模型进行预测
    pre_train = clf.predict(train_x)
    pre_valid = clf.predict(valid_x)
    pre_test = clf.predict(test_x)
    pre_new = clf.predict(new_data_x)

    # 计算准确率
    accuracy = sum(pre_train == train_y) / len(train_y)
    print("train Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_valid == valid_y) / len(valid_y)
    print("valid Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_test == test_y) / len(test_y)
    print("test Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_new == new_data_y) / len(new_data_y)
    print("new data Accuracy: {:.2f}%".format(accuracy*100))

    # 计算精确率
    precision = precision_score(train_y, pre_train)
    print("train Precision: {:.2f}%".format(precision*100))

    precision = precision_score(valid_y, pre_valid)
    print("valid Precision: {:.2f}%".format(precision*100))

    precision = precision_score(test_y, pre_test)
    print("test Precision: {:.2f}%".format(precision*100))

    precision = precision_score(new_data_y, pre_new)
    print("new data Precision: {:.2f}%".format(precision*100))

    # 计算召回率
    recall = recall_score(train_y, pre_train)
    print("train Recall: {:.2f}%".format(recall*100))

    recall = recall_score(valid_y, pre_valid)
    print("valid Recall: {:.2f}%".format(recall*100))

    recall = recall_score(test_y, pre_test)
    print("test Recall: {:.2f}%".format(recall*100))

    recall = recall_score(new_data_y, pre_new)
    print("new data Recall: {:.2f}%".format(recall*100))

    # 计算F1-score
    f1 = f1_score(train_y, pre_train)
    print("train F1-score: {:.2f}".format(f1))

    f1 = f1_score(valid_y, pre_valid)
    print("valid F1-score: {:.2f}".format(f1))

    f1 = f1_score(test_y, pre_test)
    print("test F1-score: {:.2f}".format(f1))

    f1 = f1_score(new_data_y, pre_new)
    print("new data F1-score: {:.2f}".format(f1))

    # 计算AUC-ROC
    auc = roc_auc_score(train_y, pre_train)
    print("train AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(valid_y, pre_valid)
    print("valid AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(test_y, pre_test)
    print("test AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(new_data_y, pre_new)
    print("new data AUC-ROC: {:.2f}".format(auc))

    # 计算混淆矩阵
    confusion_train = confusion_matrix(train_y, pre_train)
    print("train Confusion Matrix:")
    print(confusion_train)

    confusion_valid = confusion_matrix(valid_y, pre_valid)
    print("valid Confusion Matrix:")
    print(confusion_valid)

    confusion_test = confusion_matrix(test_y, pre_test)
    print("test Confusion Matrix:")
    print(confusion_test)

    confusion_new = confusion_matrix(new_data_y, pre_new)
    print("new data Confusion Matrix:")
    print(confusion_new)


    # 创建K近邻模型实例
    print("***********************KNeighborsClassifier(n_neighbors=3, weights='uniform')***************************")

    clf = KNeighborsClassifier(n_neighbors=3, weights="uniform")
    # 训练模型
    clf = clf.fit(train_x, train_y)

    # 使用模型进行预测
    pre_train = clf.predict(train_x)
    pre_valid = clf.predict(valid_x)
    pre_test = clf.predict(test_x)
    pre_new = clf.predict(new_data_x)

    # 计算准确率
    accuracy = sum(pre_train == train_y) / len(train_y)
    print("train Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_valid == valid_y) / len(valid_y)
    print("valid Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_test == test_y) / len(test_y)
    print("test Accuracy: {:.2f}%".format(accuracy*100))

    accuracy = sum(pre_new == new_data_y) / len(new_data_y)
    print("new data Accuracy: {:.2f}%".format(accuracy*100))

    # 计算精确率
    precision = precision_score(train_y, pre_train)
    print("train Precision: {:.2f}%".format(precision*100))

    precision = precision_score(valid_y, pre_valid)
    print("valid Precision: {:.2f}%".format(precision*100))

    precision = precision_score(test_y, pre_test)
    print("test Precision: {:.2f}%".format(precision*100))

    precision = precision_score(new_data_y, pre_new)
    print("new data Precision: {:.2f}%".format(precision*100))

    # 计算召回率
    recall = recall_score(train_y, pre_train)
    print("train Recall: {:.2f}%".format(recall*100))

    recall = recall_score(valid_y, pre_valid)
    print("valid Recall: {:.2f}%".format(recall*100))

    recall = recall_score(test_y, pre_test)
    print("test Recall: {:.2f}%".format(recall*100))

    recall = recall_score(new_data_y, pre_new)
    print("new data Recall: {:.2f}%".format(recall*100))

    # 计算F1-score
    f1 = f1_score(train_y, pre_train)
    print("train F1-score: {:.2f}".format(f1))

    f1 = f1_score(valid_y, pre_valid)
    print("valid F1-score: {:.2f}".format(f1))

    f1 = f1_score(test_y, pre_test)
    print("test F1-score: {:.2f}".format(f1))

    f1 = f1_score(new_data_y, pre_new)
    print("new data F1-score: {:.2f}".format(f1))

    # 计算AUC-ROC
    auc = roc_auc_score(train_y, pre_train)
    print("train AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(valid_y, pre_valid)
    print("valid AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(test_y, pre_test)
    print("test AUC-ROC: {:.2f}".format(auc))

    auc = roc_auc_score(new_data_y, pre_new)
    print("new data AUC-ROC: {:.2f}".format(auc))

    # 计算混淆矩阵
    confusion_train = confusion_matrix(train_y, pre_train)
    print("train Confusion Matrix:")
    print(confusion_train)

    confusion_valid = confusion_matrix(valid_y, pre_valid)
    print("valid Confusion Matrix:")
    print(confusion_valid)

    confusion_test = confusion_matrix(test_y, pre_test)
    print("test Confusion Matrix:")
    print(confusion_test)

    confusion_new = confusion_matrix(new_data_y, pre_new)
    print("new data Confusion Matrix:")
    print(confusion_new)


        