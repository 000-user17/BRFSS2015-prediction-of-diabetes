#!/bin/bash

cd /home/cs/binaryClass
python data_divide.py --path ./data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv
cd machine_learning
python tree.py > output_tree.txt
python random_forest.py > output_random_forest.txt
python k-neighbor.py > output_k-neighbor.txt
python svm.py > output_svm.txt

cd /home/cs/binaryClass
python data_divide.py --path ./data/normalized_data_binary5050.csv
cd machine_learning
python tree.py > output_tree_norm.txt
python random_forest.py > output_random_forest_norm.txt
python k-neighbor.py > output_k-neighbor_norm.txt
python svm.py > output_svm_norm.txt

cd /home/cs/binaryClass
python data_divide.py --path ./data/data_feature_select_chi2.csv
cd machine_learning
python tree.py > output_tree_chi2.txt
python random_forest.py > output_random_forest_chi2.txt
python k-neighbor.py > output_k-neighbor_chi2.txt
python svm.py > output_svm_chi2.txt

cd /home/cs/binaryClass
python data_divide.py --path ./data/data_feature_select_f_classif.csv
cd machine_learning
python tree.py > output_tree_f_classif.txt
python random_forest.py > output_random_forest_f_classif.txt
python k-neighbor.py > output_k-neighbor_f_classif.txt
python svm.py > output_svm_f_classif.txt

cd /home/cs/binaryClass
python data_divide.py --path ./data/data_feature_select_pearson.csv
cd machine_learning
python tree.py > output_tree_pearson.txt
python random_forest.py > output_random_forest_pearson.txt
python k-neighbor.py > output_k-neighbor_pearson.txt
python svm.py > output_svm_pearson.txt





