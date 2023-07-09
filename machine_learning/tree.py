import numpy as np
import pandas as pd
import sklearn
from sklearn import tree
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

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
    


    # 创建决策树模型实例
    print("*****************************tree.DecisionTreeClassifier(criterion='gini', splitter='best')************************************")
    clf = tree.DecisionTreeClassifier(criterion='gini', splitter="best")

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
    print("test Accuracy: {:.2f}%".format(accuracy*100))

    # 计算精确率、召回率和F1
    print("train classification report:\n", classification_report(train_y, pre_train))
    print("valid classification report:\n", classification_report(valid_y, pre_valid))
    print("test classification report:\n", classification_report(test_y, pre_test))
    print("new_data classification report:\n", classification_report(new_data_y, pre_new))

    # 计算AUC-RUC
    train_auc_score = roc_auc_score(train_y, clf.predict_proba(train_x)[:, 1])
    valid_auc_score = roc_auc_score(valid_y, clf.predict_proba(valid_x)[:, 1])
    test_auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
    new_data_auc_score = roc_auc_score(new_data_y, clf.predict_proba(new_data_x)[:, 1])
    print("train AUC score:", train_auc_score)
    print("valid AUC score:", valid_auc_score)
    print("test AUC score:", test_auc_score)
    print("new_data AUC score:", new_data_auc_score)

    # 计算混淆矩阵
    print("train confusion matrix:\n", confusion_matrix(train_y, pre_train))
    print("valid confusion matrix:\n", confusion_matrix(valid_y, pre_valid))
    print("test confusion matrix:\n", confusion_matrix(test_y, pre_test))
    print("new_data confusion matrix:\n", confusion_matrix(new_data_y, pre_new))



    # 创建决策树模型实例
    print("*****************************tree.DecisionTreeClassifier(criterion='entropy', splitter='best')************************************")
    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter="best")

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
    print("test Accuracy: {:.2f}%".format(accuracy*100))

    # 计算精确率、召回率和F1
    print("train classification report:\n", classification_report(train_y, pre_train))
    print("valid classification report:\n", classification_report(valid_y, pre_valid))
    print("test classification report:\n", classification_report(test_y, pre_test))
    print("new_data classification report:\n", classification_report(new_data_y, pre_new))

    # 计算AUC-RUC
    train_auc_score = roc_auc_score(train_y, clf.predict_proba(train_x)[:, 1])
    valid_auc_score = roc_auc_score(valid_y, clf.predict_proba(valid_x)[:, 1])
    test_auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
    new_data_auc_score = roc_auc_score(new_data_y, clf.predict_proba(new_data_x)[:, 1])
    print("train AUC score:", train_auc_score)
    print("valid AUC score:", valid_auc_score)
    print("test AUC score:", test_auc_score)
    print("new_data AUC score:", new_data_auc_score)

    # 计算混淆矩阵
    print("train confusion matrix:\n", confusion_matrix(train_y, pre_train))
    print("valid confusion matrix:\n", confusion_matrix(valid_y, pre_valid))
    print("test confusion matrix:\n", confusion_matrix(test_y, pre_test))
    print("new_data confusion matrix:\n", confusion_matrix(new_data_y, pre_new))


    # 创建决策树模型实例
    print("*****************************tree.DecisionTreeClassifier(criterion='gini', splitter='random')************************************")
    clf = tree.DecisionTreeClassifier(criterion='gini', splitter="random")

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
    print("test Accuracy: {:.2f}%".format(accuracy*100))

    # 计算精确率、召回率和F1
    print("train classification report:\n", classification_report(train_y, pre_train))
    print("valid classification report:\n", classification_report(valid_y, pre_valid))
    print("test classification report:\n", classification_report(test_y, pre_test))
    print("new_data classification report:\n", classification_report(new_data_y, pre_new))

    # 计算AUC-RUC
    train_auc_score = roc_auc_score(train_y, clf.predict_proba(train_x)[:, 1])
    valid_auc_score = roc_auc_score(valid_y, clf.predict_proba(valid_x)[:, 1])
    test_auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
    new_data_auc_score = roc_auc_score(new_data_y, clf.predict_proba(new_data_x)[:, 1])
    print("train AUC score:", train_auc_score)
    print("valid AUC score:", valid_auc_score)
    print("test AUC score:", test_auc_score)
    print("new_data AUC score:", new_data_auc_score)

    # 计算混淆矩阵
    print("train confusion matrix:\n", confusion_matrix(train_y, pre_train))
    print("valid confusion matrix:\n", confusion_matrix(valid_y, pre_valid))
    print("test confusion matrix:\n", confusion_matrix(test_y, pre_test))
    print("new_data confusion matrix:\n", confusion_matrix(new_data_y, pre_new))


    # 创建决策树模型实例
    print("*****************************tree.DecisionTreeClassifier(criterion='entropy', splitter='random')************************************")
    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter="random")

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
    print("test Accuracy: {:.2f}%".format(accuracy*100))

    # 计算精确率、召回率和F1
    print("train classification report:\n", classification_report(train_y, pre_train))
    print("valid classification report:\n", classification_report(valid_y, pre_valid))
    print("test classification report:\n", classification_report(test_y, pre_test))
    print("new_data classification report:\n", classification_report(new_data_y, pre_new))

    # 计算AUC-RUC
    train_auc_score = roc_auc_score(train_y, clf.predict_proba(train_x)[:, 1])
    valid_auc_score = roc_auc_score(valid_y, clf.predict_proba(valid_x)[:, 1])
    test_auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
    new_data_auc_score = roc_auc_score(new_data_y, clf.predict_proba(new_data_x)[:, 1])
    print("train AUC score:", train_auc_score)
    print("valid AUC score:", valid_auc_score)
    print("test AUC score:", test_auc_score)
    print("new_data AUC score:", new_data_auc_score)

    # 计算混淆矩阵
    print("train confusion matrix:\n", confusion_matrix(train_y, pre_train))
    print("valid confusion matrix:\n", confusion_matrix(valid_y, pre_valid))
    print("test confusion matrix:\n", confusion_matrix(test_y, pre_test))
    print("new_data confusion matrix:\n", confusion_matrix(new_data_y, pre_new))