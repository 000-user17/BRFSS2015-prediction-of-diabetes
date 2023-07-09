#!/bin/bash

cd /home/cs/binaryClass
python data_divide.py --path ./data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv
cd deep_learning
python mlp.py > output_mlp.txt

cd /home/cs/binaryClass
python data_divide.py --path ./data/normalized_data_binary5050.csv
cd deep_learning
python mlp.py > output_mlp_norm.txt

cd /home/cs/binaryClass
python data_divide.py --path ./data/data_feature_select_chi2.csv
cd deep_learning
python mlp.py > output_mlp_chi2.txt

cd /home/cs/binaryClass
python data_divide.py --path ./data/data_feature_select_f_classif.csv
cd deep_learning
python mlp.py > output_mlp_f_classif.txt

cd /home/cs/binaryClass
python data_divide.py --path ./data/data_feature_select_pearson.csv
cd deep_learning
python mlp.py > output_mlp_pearson.txt





