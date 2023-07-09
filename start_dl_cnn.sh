#!/bin/bash

cd /home/cs/binaryClass
python data_divide.py --path ./data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv
cd deep_learning
python CNN.py > output_CNN.txt

cd /home/cs/binaryClass
python data_divide.py --path ./data/normalized_data_binary5050.csv
cd deep_learning
python CNN.py > output_CNN_norm.txt

cd /home/cs/binaryClass
python data_divide.py --path ./data/data_feature_select_chi2.csv
cd deep_learning
python CNN.py > output_CNN_chi2.txt

cd /home/cs/binaryClass
python data_divide.py --path ./data/data_feature_select_f_classif.csv
cd deep_learning
python CNN.py > output_CNN_f_classif.txt

cd /home/cs/binaryClass
python data_divide.py --path ./data/data_feature_select_pearson.csv
cd deep_learning
python CNN.py > output_CNN_pearson.txt





