import numpy as np
import pandas as pd
import sklearn
from sklearn import svm, metrics
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
    
 

    # 使用SVM模型进行训练
    print("*********************svm.SVC(kernel='linear', C=1, verbose=False, max_iter=1000)********************")
    clf = svm.SVC(kernel='linear', C=1, verbose=False, max_iter=1000)
    clf.fit(train_x, train_y)

    # 使用模型进行预测
    pre_train = clf.predict(train_x)
    pre_valid = clf.predict(valid_x)
    pre_test = clf.predict(test_x)
    pre_new = clf.predict(new_data_x)

    # 计算准确率
    accuracy_train = metrics.accuracy_score(train_y, pre_train)
    print("train Accuracy: {:.2f}%".format(accuracy_train * 100))

    accuracy_valid = metrics.accuracy_score(valid_y, pre_valid)
    print("valid Accuracy: {:.2f}%".format(accuracy_valid * 100))

    accuracy_test = metrics.accuracy_score(test_y, pre_test)
    print("test Accuracy: {:.2f}%".format(accuracy_test * 100))

    accuracy_new = metrics.accuracy_score(new_data_y, pre_new)
    print("new data Accuracy: {:.2f}%".format(accuracy_new * 100))

    # 计算精确率
    precision_train = metrics.precision_score(train_y, pre_train)
    print("train Precision: {:.2f}%".format(precision_train * 100))

    precision_valid = metrics.precision_score(valid_y, pre_valid)
    print("valid Precision: {:.2f}%".format(precision_valid * 100))

    precision_test = metrics.precision_score(test_y, pre_test)
    print("test Precision: {:.2f}%".format(precision_test * 100))

    precision_new = metrics.precision_score(new_data_y, pre_new)
    print("new data Precision: {:.2f}%".format(precision_new * 100))

    # 计算召回率
    recall_train = metrics.recall_score(train_y, pre_train)
    print("train Recall: {:.2f}%".format(recall_train * 100))

    recall_valid = metrics.recall_score(valid_y, pre_valid)
    print("valid Recall: {:.2f}%".format(recall_valid * 100))

    recall_test = metrics.recall_score(test_y, pre_test)
    print("test Recall: {:.2f}%".format(recall_test * 100))

    recall_new = metrics.recall_score(new_data_y, pre_new)
    print("new data Recall: {:.2f}%".format(recall_new * 100))

    # 计算F1-score
    f1_score_train = metrics.f1_score(train_y, pre_train)
    print("train F1-score: {:.2f}".format(f1_score_train))

    f1_score_valid = metrics.f1_score(valid_y, pre_valid)
    print("valid F1-score: {:.2f}".format(f1_score_valid))

    f1_score_test = metrics.f1_score(test_y, pre_test)
    print("test F1-score: {:.2f}".format(f1_score_test))

    f1_score_new = metrics.f1_score(new_data_y, pre_new)
    print("new data F1-score: {:.2f}".format(f1_score_new))

    # 计算AUC-ROC
    auc_train = metrics.roc_auc_score(train_y, clf.decision_function(train_x))
    print("train AUC-ROC: {:.2f}".format(auc_train))

    auc_valid = metrics.roc_auc_score(valid_y, clf.decision_function(valid_x))
    print("valid AUC-ROC: {:.2f}".format(auc_valid))

    auc_test = metrics.roc_auc_score(test_y, clf.decision_function(test_x))
    print("test AUC-ROC: {:.2f}".format(auc_test))

    auc_new = metrics.roc_auc_score(new_data_y, clf.decision_function(new_data_x))
    print("new data AUC-ROC: {:.2f}".format(auc_new))


    # 计算训练集混淆矩阵
    confusion_train = confusion_matrix(train_y, pre_train)
    print("train Confusion Matrix:")
    print(confusion_train)

    # 计算验证集混淆矩阵
    confusion_valid = confusion_matrix(valid_y, pre_valid)
    print("valid Confusion Matrix:")
    print(confusion_valid)

    # 计算测试集混淆矩阵
    confusion_test = confusion_matrix(test_y, pre_test)
    print("test Confusion Matrix:")
    print(confusion_test)

    # 计算新数据集混淆矩阵
    confusion_new = confusion_matrix(new_data_y, pre_new)
    print("new data Confusion Matrix:")
    print(confusion_new)


    # 使用SVM模型进行训练
    print("*********************svm.SVC(kernel='linear', C=1, verbose=False, max_iter=10000)********************")
    clf = svm.SVC(kernel='linear', C=1, verbose=False, max_iter=10000)
    clf.fit(train_x, train_y)

    # 使用模型进行预测
    pre_train = clf.predict(train_x)
    pre_valid = clf.predict(valid_x)
    pre_test = clf.predict(test_x)
    pre_new = clf.predict(new_data_x)

    # 计算准确率
    accuracy_train = metrics.accuracy_score(train_y, pre_train)
    print("train Accuracy: {:.2f}%".format(accuracy_train * 100))

    accuracy_valid = metrics.accuracy_score(valid_y, pre_valid)
    print("valid Accuracy: {:.2f}%".format(accuracy_valid * 100))

    accuracy_test = metrics.accuracy_score(test_y, pre_test)
    print("test Accuracy: {:.2f}%".format(accuracy_test * 100))

    accuracy_new = metrics.accuracy_score(new_data_y, pre_new)
    print("new data Accuracy: {:.2f}%".format(accuracy_new * 100))

    # 计算精确率
    precision_train = metrics.precision_score(train_y, pre_train)
    print("train Precision: {:.2f}%".format(precision_train * 100))

    precision_valid = metrics.precision_score(valid_y, pre_valid)
    print("valid Precision: {:.2f}%".format(precision_valid * 100))

    precision_test = metrics.precision_score(test_y, pre_test)
    print("test Precision: {:.2f}%".format(precision_test * 100))

    precision_new = metrics.precision_score(new_data_y, pre_new)
    print("new data Precision: {:.2f}%".format(precision_new * 100))

    # 计算召回率
    recall_train = metrics.recall_score(train_y, pre_train)
    print("train Recall: {:.2f}%".format(recall_train * 100))

    recall_valid = metrics.recall_score(valid_y, pre_valid)
    print("valid Recall: {:.2f}%".format(recall_valid * 100))

    recall_test = metrics.recall_score(test_y, pre_test)
    print("test Recall: {:.2f}%".format(recall_test * 100))

    recall_new = metrics.recall_score(new_data_y, pre_new)
    print("new data Recall: {:.2f}%".format(recall_new * 100))

    # 计算F1-score
    f1_score_train = metrics.f1_score(train_y, pre_train)
    print("train F1-score: {:.2f}".format(f1_score_train))

    f1_score_valid = metrics.f1_score(valid_y, pre_valid)
    print("valid F1-score: {:.2f}".format(f1_score_valid))

    f1_score_test = metrics.f1_score(test_y, pre_test)
    print("test F1-score: {:.2f}".format(f1_score_test))

    f1_score_new = metrics.f1_score(new_data_y, pre_new)
    print("new data F1-score: {:.2f}".format(f1_score_new))

    # 计算AUC-ROC
    auc_train = metrics.roc_auc_score(train_y, clf.decision_function(train_x))
    print("train AUC-ROC: {:.2f}".format(auc_train))

    auc_valid = metrics.roc_auc_score(valid_y, clf.decision_function(valid_x))
    print("valid AUC-ROC: {:.2f}".format(auc_valid))

    auc_test = metrics.roc_auc_score(test_y, clf.decision_function(test_x))
    print("test AUC-ROC: {:.2f}".format(auc_test))

    auc_new = metrics.roc_auc_score(new_data_y, clf.decision_function(new_data_x))
    print("new data AUC-ROC: {:.2f}".format(auc_new))


    # 计算训练集混淆矩阵
    confusion_train = confusion_matrix(train_y, pre_train)
    print("train Confusion Matrix:")
    print(confusion_train)

    # 计算验证集混淆矩阵
    confusion_valid = confusion_matrix(valid_y, pre_valid)
    print("valid Confusion Matrix:")
    print(confusion_valid)

    # 计算测试集混淆矩阵
    confusion_test = confusion_matrix(test_y, pre_test)
    print("test Confusion Matrix:")
    print(confusion_test)

    # 计算新数据集混淆矩阵
    confusion_new = confusion_matrix(new_data_y, pre_new)
    print("new data Confusion Matrix:")
    print(confusion_new)


    # 使用SVM模型进行训练
    print("*********************svm.SVC(kernel='linear', C=1, verbose=False, max_iter=50000)********************")
    clf = svm.SVC(kernel='linear', C=1, verbose=False, max_iter=50000)
    clf.fit(train_x, train_y)

    # 使用模型进行预测
    pre_train = clf.predict(train_x)
    pre_valid = clf.predict(valid_x)
    pre_test = clf.predict(test_x)
    pre_new = clf.predict(new_data_x)
    # 计算准确率
    accuracy_train = metrics.accuracy_score(train_y, pre_train)
    print("train Accuracy: {:.2f}%".format(accuracy_train * 100))

    accuracy_valid = metrics.accuracy_score(valid_y, pre_valid)
    print("valid Accuracy: {:.2f}%".format(accuracy_valid * 100))

    accuracy_test = metrics.accuracy_score(test_y, pre_test)
    print("test Accuracy: {:.2f}%".format(accuracy_test * 100))

    accuracy_new = metrics.accuracy_score(new_data_y, pre_new)
    print("new data Accuracy: {:.2f}%".format(accuracy_new * 100))

    # 计算精确率
    precision_train = metrics.precision_score(train_y, pre_train)
    print("train Precision: {:.2f}%".format(precision_train * 100))

    precision_valid = metrics.precision_score(valid_y, pre_valid)
    print("valid Precision: {:.2f}%".format(precision_valid * 100))

    precision_test = metrics.precision_score(test_y, pre_test)
    print("test Precision: {:.2f}%".format(precision_test * 100))

    precision_new = metrics.precision_score(new_data_y, pre_new)
    print("new data Precision: {:.2f}%".format(precision_new * 100))

    # 计算召回率
    recall_train = metrics.recall_score(train_y, pre_train)
    print("train Recall: {:.2f}%".format(recall_train * 100))

    recall_valid = metrics.recall_score(valid_y, pre_valid)
    print("valid Recall: {:.2f}%".format(recall_valid * 100))

    recall_test = metrics.recall_score(test_y, pre_test)
    print("test Recall: {:.2f}%".format(recall_test * 100))

    recall_new = metrics.recall_score(new_data_y, pre_new)
    print("new data Recall: {:.2f}%".format(recall_new * 100))

    # 计算F1-score
    f1_score_train = metrics.f1_score(train_y, pre_train)
    print("train F1-score: {:.2f}".format(f1_score_train))

    f1_score_valid = metrics.f1_score(valid_y, pre_valid)
    print("valid F1-score: {:.2f}".format(f1_score_valid))

    f1_score_test = metrics.f1_score(test_y, pre_test)
    print("test F1-score: {:.2f}".format(f1_score_test))

    f1_score_new = metrics.f1_score(new_data_y, pre_new)
    print("new data F1-score: {:.2f}".format(f1_score_new))

    # 计算AUC-ROC
    auc_train = metrics.roc_auc_score(train_y, clf.decision_function(train_x))
    print("train AUC-ROC: {:.2f}".format(auc_train))

    auc_valid = metrics.roc_auc_score(valid_y, clf.decision_function(valid_x))
    print("valid AUC-ROC: {:.2f}".format(auc_valid))

    auc_test = metrics.roc_auc_score(test_y, clf.decision_function(test_x))
    print("test AUC-ROC: {:.2f}".format(auc_test))

    auc_new = metrics.roc_auc_score(new_data_y, clf.decision_function(new_data_x))
    print("new data AUC-ROC: {:.2f}".format(auc_new))


    # 计算训练集混淆矩阵
    confusion_train = confusion_matrix(train_y, pre_train)
    print("train Confusion Matrix:")
    print(confusion_train)

    # 计算验证集混淆矩阵
    confusion_valid = confusion_matrix(valid_y, pre_valid)
    print("valid Confusion Matrix:")
    print(confusion_valid)

    # 计算测试集混淆矩阵
    confusion_test = confusion_matrix(test_y, pre_test)
    print("test Confusion Matrix:")
    print(confusion_test)

    # 计算新数据集混淆矩阵
    confusion_new = confusion_matrix(new_data_y, pre_new)
    print("new data Confusion Matrix:")
    print(confusion_new)


    # 使用SVM模型进行训练
    print("*********************svm.SVC(kernel='linear', C=2, verbose=False, max_iter=1000)********************")
    clf = svm.SVC(kernel='linear', C=2, verbose=False, max_iter=1000)
    clf.fit(train_x, train_y)

    # 使用模型进行预测
    pre_train = clf.predict(train_x)
    pre_valid = clf.predict(valid_x)
    pre_test = clf.predict(test_x)
    pre_new = clf.predict(new_data_x)
    # 计算准确率
    accuracy_train = metrics.accuracy_score(train_y, pre_train)
    print("train Accuracy: {:.2f}%".format(accuracy_train * 100))

    accuracy_valid = metrics.accuracy_score(valid_y, pre_valid)
    print("valid Accuracy: {:.2f}%".format(accuracy_valid * 100))

    accuracy_test = metrics.accuracy_score(test_y, pre_test)
    print("test Accuracy: {:.2f}%".format(accuracy_test * 100))

    accuracy_new = metrics.accuracy_score(new_data_y, pre_new)
    print("new data Accuracy: {:.2f}%".format(accuracy_new * 100))

    # 计算精确率
    precision_train = metrics.precision_score(train_y, pre_train)
    print("train Precision: {:.2f}%".format(precision_train * 100))

    precision_valid = metrics.precision_score(valid_y, pre_valid)
    print("valid Precision: {:.2f}%".format(precision_valid * 100))

    precision_test = metrics.precision_score(test_y, pre_test)
    print("test Precision: {:.2f}%".format(precision_test * 100))

    precision_new = metrics.precision_score(new_data_y, pre_new)
    print("new data Precision: {:.2f}%".format(precision_new * 100))

    # 计算召回率
    recall_train = metrics.recall_score(train_y, pre_train)
    print("train Recall: {:.2f}%".format(recall_train * 100))

    recall_valid = metrics.recall_score(valid_y, pre_valid)
    print("valid Recall: {:.2f}%".format(recall_valid * 100))

    recall_test = metrics.recall_score(test_y, pre_test)
    print("test Recall: {:.2f}%".format(recall_test * 100))

    recall_new = metrics.recall_score(new_data_y, pre_new)
    print("new data Recall: {:.2f}%".format(recall_new * 100))

    # 计算F1-score
    f1_score_train = metrics.f1_score(train_y, pre_train)
    print("train F1-score: {:.2f}".format(f1_score_train))

    f1_score_valid = metrics.f1_score(valid_y, pre_valid)
    print("valid F1-score: {:.2f}".format(f1_score_valid))

    f1_score_test = metrics.f1_score(test_y, pre_test)
    print("test F1-score: {:.2f}".format(f1_score_test))

    f1_score_new = metrics.f1_score(new_data_y, pre_new)
    print("new data F1-score: {:.2f}".format(f1_score_new))

    # 计算AUC-ROC
    auc_train = metrics.roc_auc_score(train_y, clf.decision_function(train_x))
    print("train AUC-ROC: {:.2f}".format(auc_train))

    auc_valid = metrics.roc_auc_score(valid_y, clf.decision_function(valid_x))
    print("valid AUC-ROC: {:.2f}".format(auc_valid))

    auc_test = metrics.roc_auc_score(test_y, clf.decision_function(test_x))
    print("test AUC-ROC: {:.2f}".format(auc_test))

    auc_new = metrics.roc_auc_score(new_data_y, clf.decision_function(new_data_x))
    print("new data AUC-ROC: {:.2f}".format(auc_new))

    # 计算训练集混淆矩阵
    confusion_train = confusion_matrix(train_y, pre_train)
    print("train Confusion Matrix:")
    print(confusion_train)

    # 计算验证集混淆矩阵
    confusion_valid = confusion_matrix(valid_y, pre_valid)
    print("valid Confusion Matrix:")
    print(confusion_valid)

    # 计算测试集混淆矩阵
    confusion_test = confusion_matrix(test_y, pre_test)
    print("test Confusion Matrix:")
    print(confusion_test)

    # 计算新数据集混淆矩阵
    confusion_new = confusion_matrix(new_data_y, pre_new)
    print("new data Confusion Matrix:")
    print(confusion_new)


    # 使用SVM模型进行训练
    print("*********************svm.SVC(kernel='linear', C=2, verbose=False, max_iter=10000)********************")
    clf = svm.SVC(kernel='linear', C=2, verbose=False, max_iter=10000)
    clf.fit(train_x, train_y)

    # 使用模型进行预测
    pre_train = clf.predict(train_x)
    pre_valid = clf.predict(valid_x)
    pre_test = clf.predict(test_x)
    pre_new = clf.predict(new_data_x)

    # 计算准确率
    accuracy_train = metrics.accuracy_score(train_y, pre_train)
    print("train Accuracy: {:.2f}%".format(accuracy_train * 100))

    accuracy_valid = metrics.accuracy_score(valid_y, pre_valid)
    print("valid Accuracy: {:.2f}%".format(accuracy_valid * 100))

    accuracy_test = metrics.accuracy_score(test_y, pre_test)
    print("test Accuracy: {:.2f}%".format(accuracy_test * 100))

    accuracy_new = metrics.accuracy_score(new_data_y, pre_new)
    print("new data Accuracy: {:.2f}%".format(accuracy_new * 100))

    # 计算精确率
    precision_train = metrics.precision_score(train_y, pre_train)
    print("train Precision: {:.2f}%".format(precision_train * 100))

    precision_valid = metrics.precision_score(valid_y, pre_valid)
    print("valid Precision: {:.2f}%".format(precision_valid * 100))

    precision_test = metrics.precision_score(test_y, pre_test)
    print("test Precision: {:.2f}%".format(precision_test * 100))

    precision_new = metrics.precision_score(new_data_y, pre_new)
    print("new data Precision: {:.2f}%".format(precision_new * 100))

    # 计算召回率
    recall_train = metrics.recall_score(train_y, pre_train)
    print("train Recall: {:.2f}%".format(recall_train * 100))

    recall_valid = metrics.recall_score(valid_y, pre_valid)
    print("valid Recall: {:.2f}%".format(recall_valid * 100))

    recall_test = metrics.recall_score(test_y, pre_test)
    print("test Recall: {:.2f}%".format(recall_test * 100))

    recall_new = metrics.recall_score(new_data_y, pre_new)
    print("new data Recall: {:.2f}%".format(recall_new * 100))

    # 计算F1-score
    f1_score_train = metrics.f1_score(train_y, pre_train)
    print("train F1-score: {:.2f}".format(f1_score_train))

    f1_score_valid = metrics.f1_score(valid_y, pre_valid)
    print("valid F1-score: {:.2f}".format(f1_score_valid))

    f1_score_test = metrics.f1_score(test_y, pre_test)
    print("test F1-score: {:.2f}".format(f1_score_test))

    f1_score_new = metrics.f1_score(new_data_y, pre_new)
    print("new data F1-score: {:.2f}".format(f1_score_new))

    # 计算AUC-ROC
    auc_train = metrics.roc_auc_score(train_y, clf.decision_function(train_x))
    print("train AUC-ROC: {:.2f}".format(auc_train))

    auc_valid = metrics.roc_auc_score(valid_y, clf.decision_function(valid_x))
    print("valid AUC-ROC: {:.2f}".format(auc_valid))

    auc_test = metrics.roc_auc_score(test_y, clf.decision_function(test_x))
    print("test AUC-ROC: {:.2f}".format(auc_test))

    auc_new = metrics.roc_auc_score(new_data_y, clf.decision_function(new_data_x))
    print("new data AUC-ROC: {:.2f}".format(auc_new))


    # 计算训练集混淆矩阵
    confusion_train = confusion_matrix(train_y, pre_train)
    print("train Confusion Matrix:")
    print(confusion_train)

    # 计算验证集混淆矩阵
    confusion_valid = confusion_matrix(valid_y, pre_valid)
    print("valid Confusion Matrix:")
    print(confusion_valid)

    # 计算测试集混淆矩阵
    confusion_test = confusion_matrix(test_y, pre_test)
    print("test Confusion Matrix:")
    print(confusion_test)

    # 计算新数据集混淆矩阵
    confusion_new = confusion_matrix(new_data_y, pre_new)
    print("new data Confusion Matrix:")
    print(confusion_new)


    # 使用SVM模型进行训练
    print("*********************svm.SVC(kernel='linear', C=2, verbose=False, max_iter=50000)********************")
    clf = svm.SVC(kernel='linear', C=2, verbose=False, max_iter=50000)
    clf.fit(train_x, train_y)

    # 使用模型进行预测
    pre_train = clf.predict(train_x)
    pre_valid = clf.predict(valid_x)
    pre_test = clf.predict(test_x)
    pre_new = clf.predict(new_data_x)

    # 计算准确率
    accuracy_train = metrics.accuracy_score(train_y, pre_train)
    print("train Accuracy: {:.2f}%".format(accuracy_train * 100))

    accuracy_valid = metrics.accuracy_score(valid_y, pre_valid)
    print("valid Accuracy: {:.2f}%".format(accuracy_valid * 100))

    accuracy_test = metrics.accuracy_score(test_y, pre_test)
    print("test Accuracy: {:.2f}%".format(accuracy_test * 100))

    accuracy_new = metrics.accuracy_score(new_data_y, pre_new)
    print("new data Accuracy: {:.2f}%".format(accuracy_new * 100))

    # 计算精确率
    precision_train = metrics.precision_score(train_y, pre_train)
    print("train Precision: {:.2f}%".format(precision_train * 100))

    precision_valid = metrics.precision_score(valid_y, pre_valid)
    print("valid Precision: {:.2f}%".format(precision_valid * 100))

    precision_test = metrics.precision_score(test_y, pre_test)
    print("test Precision: {:.2f}%".format(precision_test * 100))

    precision_new = metrics.precision_score(new_data_y, pre_new)
    print("new data Precision: {:.2f}%".format(precision_new * 100))

    # 计算召回率
    recall_train = metrics.recall_score(train_y, pre_train)
    print("train Recall: {:.2f}%".format(recall_train * 100))

    recall_valid = metrics.recall_score(valid_y, pre_valid)
    print("valid Recall: {:.2f}%".format(recall_valid * 100))

    recall_test = metrics.recall_score(test_y, pre_test)
    print("test Recall: {:.2f}%".format(recall_test * 100))

    recall_new = metrics.recall_score(new_data_y, pre_new)
    print("new data Recall: {:.2f}%".format(recall_new * 100))

    # 计算F1-score
    f1_score_train = metrics.f1_score(train_y, pre_train)
    print("train F1-score: {:.2f}".format(f1_score_train))

    f1_score_valid = metrics.f1_score(valid_y, pre_valid)
    print("valid F1-score: {:.2f}".format(f1_score_valid))

    f1_score_test = metrics.f1_score(test_y, pre_test)
    print("test F1-score: {:.2f}".format(f1_score_test))

    f1_score_new = metrics.f1_score(new_data_y, pre_new)
    print("new data F1-score: {:.2f}".format(f1_score_new))

    # 计算AUC-ROC
    auc_train = metrics.roc_auc_score(train_y, clf.decision_function(train_x))
    print("train AUC-ROC: {:.2f}".format(auc_train))

    auc_valid = metrics.roc_auc_score(valid_y, clf.decision_function(valid_x))
    print("valid AUC-ROC: {:.2f}".format(auc_valid))

    auc_test = metrics.roc_auc_score(test_y, clf.decision_function(test_x))
    print("test AUC-ROC: {:.2f}".format(auc_test))

    auc_new = metrics.roc_auc_score(new_data_y, clf.decision_function(new_data_x))
    print("new data AUC-ROC: {:.2f}".format(auc_new))

    from sklearn.metrics import confusion_matrix

    # 计算训练集混淆矩阵
    confusion_train = confusion_matrix(train_y, pre_train)
    print("train Confusion Matrix:")
    print(confusion_train)

    # 计算验证集混淆矩阵
    confusion_valid = confusion_matrix(valid_y, pre_valid)
    print("valid Confusion Matrix:")
    print(confusion_valid)

    # 计算测试集混淆矩阵
    confusion_test = confusion_matrix(test_y, pre_test)
    print("test Confusion Matrix:")
    print(confusion_test)

    # 计算新数据集混淆矩阵
    confusion_new = confusion_matrix(new_data_y, pre_new)
    print("new data Confusion Matrix:")
    print(confusion_new)


    # 使用SVM模型进行训练
    print("*********************svm.SVC(kernel='poly', C=1, verbose=False, max_iter=10000)********************")
    clf = svm.SVC( C=1, verbose=False, max_iter=10000, kernel="poly")
    clf.fit(train_x, train_y)

    # 使用模型进行预测
    pre_train = clf.predict(train_x)
    pre_valid = clf.predict(valid_x)
    pre_test = clf.predict(test_x)
    pre_new = clf.predict(new_data_x)

    # 计算准确率
    accuracy_train = metrics.accuracy_score(train_y, pre_train)
    print("train Accuracy: {:.2f}%".format(accuracy_train * 100))

    accuracy_valid = metrics.accuracy_score(valid_y, pre_valid)
    print("valid Accuracy: {:.2f}%".format(accuracy_valid * 100))

    accuracy_test = metrics.accuracy_score(test_y, pre_test)
    print("test Accuracy: {:.2f}%".format(accuracy_test * 100))

    accuracy_new = metrics.accuracy_score(new_data_y, pre_new)
    print("new data Accuracy: {:.2f}%".format(accuracy_new * 100))

    # 计算精确率
    precision_train = metrics.precision_score(train_y, pre_train)
    print("train Precision: {:.2f}%".format(precision_train * 100))

    precision_valid = metrics.precision_score(valid_y, pre_valid)
    print("valid Precision: {:.2f}%".format(precision_valid * 100))

    precision_test = metrics.precision_score(test_y, pre_test)
    print("test Precision: {:.2f}%".format(precision_test * 100))

    precision_new = metrics.precision_score(new_data_y, pre_new)
    print("new data Precision: {:.2f}%".format(precision_new * 100))

    # 计算召回率
    recall_train = metrics.recall_score(train_y, pre_train)
    print("train Recall: {:.2f}%".format(recall_train * 100))

    recall_valid = metrics.recall_score(valid_y, pre_valid)
    print("valid Recall: {:.2f}%".format(recall_valid * 100))

    recall_test = metrics.recall_score(test_y, pre_test)
    print("test Recall: {:.2f}%".format(recall_test * 100))

    recall_new = metrics.recall_score(new_data_y, pre_new)
    print("new data Recall: {:.2f}%".format(recall_new * 100))

    # 计算F1-score
    f1_score_train = metrics.f1_score(train_y, pre_train)
    print("train F1-score: {:.2f}".format(f1_score_train))

    f1_score_valid = metrics.f1_score(valid_y, pre_valid)
    print("valid F1-score: {:.2f}".format(f1_score_valid))

    f1_score_test = metrics.f1_score(test_y, pre_test)
    print("test F1-score: {:.2f}".format(f1_score_test))

    f1_score_new = metrics.f1_score(new_data_y, pre_new)
    print("new data F1-score: {:.2f}".format(f1_score_new))

    # 计算AUC-ROC
    auc_train = metrics.roc_auc_score(train_y, clf.decision_function(train_x))
    print("train AUC-ROC: {:.2f}".format(auc_train))

    auc_valid = metrics.roc_auc_score(valid_y, clf.decision_function(valid_x))
    print("valid AUC-ROC: {:.2f}".format(auc_valid))

    auc_test = metrics.roc_auc_score(test_y, clf.decision_function(test_x))
    print("test AUC-ROC: {:.2f}".format(auc_test))

    auc_new = metrics.roc_auc_score(new_data_y, clf.decision_function(new_data_x))
    print("new data AUC-ROC: {:.2f}".format(auc_new))


    # 计算训练集混淆矩阵
    confusion_train = confusion_matrix(train_y, pre_train)
    print("train Confusion Matrix:")
    print(confusion_train)

    # 计算验证集混淆矩阵
    confusion_valid = confusion_matrix(valid_y, pre_valid)
    print("valid Confusion Matrix:")
    print(confusion_valid)

    # 计算测试集混淆矩阵
    confusion_test = confusion_matrix(test_y, pre_test)
    print("test Confusion Matrix:")
    print(confusion_test)

    # 计算新数据集混淆矩阵
    confusion_new = confusion_matrix(new_data_y, pre_new)
    print("new data Confusion Matrix:")
    print(confusion_new)


    # 使用SVM模型进行训练
    print("*********************svm.SVC(kernel='sigmoid', C=1, verbose=False, max_iter=10000)********************")
    clf = svm.SVC(C=1, verbose=False, max_iter=10000, kernel="sigmoid")
    clf.fit(train_x, train_y)

    # 使用模型进行预测
    pre_train = clf.predict(train_x)
    pre_valid = clf.predict(valid_x)
    pre_test = clf.predict(test_x)
    pre_new = clf.predict(new_data_x)

    # 计算准确率
    accuracy_train = metrics.accuracy_score(train_y, pre_train)
    print("train Accuracy: {:.2f}%".format(accuracy_train * 100))

    accuracy_valid = metrics.accuracy_score(valid_y, pre_valid)
    print("valid Accuracy: {:.2f}%".format(accuracy_valid * 100))

    accuracy_test = metrics.accuracy_score(test_y, pre_test)
    print("test Accuracy: {:.2f}%".format(accuracy_test * 100))

    accuracy_new = metrics.accuracy_score(new_data_y, pre_new)
    print("new data Accuracy: {:.2f}%".format(accuracy_new * 100))

    # 计算精确率
    precision_train = metrics.precision_score(train_y, pre_train)
    print("train Precision: {:.2f}%".format(precision_train * 100))

    precision_valid = metrics.precision_score(valid_y, pre_valid)
    print("valid Precision: {:.2f}%".format(precision_valid * 100))

    precision_test = metrics.precision_score(test_y, pre_test)
    print("test Precision: {:.2f}%".format(precision_test * 100))

    precision_new = metrics.precision_score(new_data_y, pre_new)
    print("new data Precision: {:.2f}%".format(precision_new * 100))

    # 计算召回率
    recall_train = metrics.recall_score(train_y, pre_train)
    print("train Recall: {:.2f}%".format(recall_train * 100))

    recall_valid = metrics.recall_score(valid_y, pre_valid)
    print("valid Recall: {:.2f}%".format(recall_valid * 100))

    recall_test = metrics.recall_score(test_y, pre_test)
    print("test Recall: {:.2f}%".format(recall_test * 100))

    recall_new = metrics.recall_score(new_data_y, pre_new)
    print("new data Recall: {:.2f}%".format(recall_new * 100))

    # 计算F1-score
    f1_score_train = metrics.f1_score(train_y, pre_train)
    print("train F1-score: {:.2f}".format(f1_score_train))

    f1_score_valid = metrics.f1_score(valid_y, pre_valid)
    print("valid F1-score: {:.2f}".format(f1_score_valid))

    f1_score_test = metrics.f1_score(test_y, pre_test)
    print("test F1-score: {:.2f}".format(f1_score_test))

    f1_score_new = metrics.f1_score(new_data_y, pre_new)
    print("new data F1-score: {:.2f}".format(f1_score_new))

    # 计算AUC-ROC
    auc_train = metrics.roc_auc_score(train_y, clf.decision_function(train_x))
    print("train AUC-ROC: {:.2f}".format(auc_train))

    auc_valid = metrics.roc_auc_score(valid_y, clf.decision_function(valid_x))
    print("valid AUC-ROC: {:.2f}".format(auc_valid))

    auc_test = metrics.roc_auc_score(test_y, clf.decision_function(test_x))
    print("test AUC-ROC: {:.2f}".format(auc_test))

    auc_new = metrics.roc_auc_score(new_data_y, clf.decision_function(new_data_x))
    print("new data AUC-ROC: {:.2f}".format(auc_new))


    # 计算训练集混淆矩阵
    confusion_train = confusion_matrix(train_y, pre_train)
    print("train Confusion Matrix:")
    print(confusion_train)

    # 计算验证集混淆矩阵
    confusion_valid = confusion_matrix(valid_y, pre_valid)
    print("valid Confusion Matrix:")
    print(confusion_valid)

    # 计算测试集混淆矩阵
    confusion_test = confusion_matrix(test_y, pre_test)
    print("test Confusion Matrix:")
    print(confusion_test)

    # 计算新数据集混淆矩阵
    confusion_new = confusion_matrix(new_data_y, pre_new)
    print("new data Confusion Matrix:")
    print(confusion_new)


