#imports
'''原始数据网址：https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system?select=2015.csv
下载后放入./data  目录下，执行该脚本 获得清洗后的干净糖尿病数据'''
import os
import pandas as pd
import numpy as np
import random
random.seed(1)


if __name__ == "__main__":
    #read in the dataset (select 2015)
    year = '2015'
    path = f"./data/behavioral-risk-factor-surveillance-system/{year}.csv'"  #原始未处理的文件所在位置
    brfss_2015_dataset = pd.read_csv(path)
    
    #check that the data loaded in is in the correct format
    pd.set_option('display.max_columns', 500)
    brfss_2015_dataset.head()
    
    # select specific columns
    brfss_df_selected = brfss_2015_dataset[['DIABETE3',
                                         '_RFHYPE5',  
                                         'TOLDHI2', '_CHOLCHK', 
                                         '_BMI5', 
                                         'SMOKE100', 
                                         'CVDSTRK3', '_MICHD', 
                                         '_TOTINDA', 
                                         '_FRTLT1', '_VEGLT1', 
                                         '_RFDRHV5', 
                                         'HLTHPLN1', 'MEDCOST', 
                                         'GENHLTH', 'MENTHLTH', 'PHYSHLTH', 'DIFFWALK', 
                                         'SEX', '_AGEG5YR', 'EDUCA', 'INCOME2' ]]
    
    #Drop Missing Values - knocks 100,000 rows out right away
    brfss_df_selected = brfss_df_selected.dropna()
    brfss_df_selected.shape
    
    # DIABETE3
    # going to make this ordinal. 0 is for no diabetes or only during pregnancy, 1 is for pre-diabetes or borderline diabetes, 2 is for yes diabetes
    # Remove all 7 (dont knows)
    # Remove all 9 (refused)
    brfss_df_selected['DIABETE3'] = brfss_df_selected['DIABETE3'].replace({2:0, 3:0, 1:2, 4:1})
    brfss_df_selected = brfss_df_selected[brfss_df_selected.DIABETE3 != 7]
    brfss_df_selected = brfss_df_selected[brfss_df_selected.DIABETE3 != 9]
    brfss_df_selected.DIABETE3.unique()
    
    #1 _RFHYPE5
    #Change 1 to 0 so it represetnts No high blood pressure and 2 to 1 so it represents high blood pressure
    brfss_df_selected['_RFHYPE5'] = brfss_df_selected['_RFHYPE5'].replace({1:0, 2:1})
    brfss_df_selected = brfss_df_selected[brfss_df_selected._RFHYPE5 != 9]
    brfss_df_selected._RFHYPE5.unique()
    
    #2 TOLDHI2
    # Change 2 to 0 because it is No
    # Remove all 7 (dont knows)
    # Remove all 9 (refused)
    brfss_df_selected['TOLDHI2'] = brfss_df_selected['TOLDHI2'].replace({2:0})
    brfss_df_selected = brfss_df_selected[brfss_df_selected.TOLDHI2 != 7]
    brfss_df_selected = brfss_df_selected[brfss_df_selected.TOLDHI2 != 9]
    brfss_df_selected.TOLDHI2.unique()
    
    #3 _CHOLCHK
    # Change 3 to 0 and 2 to 0 for Not checked cholesterol in past 5 years
    # Remove 9
    brfss_df_selected['_CHOLCHK'] = brfss_df_selected['_CHOLCHK'].replace({3:0,2:0})
    brfss_df_selected = brfss_df_selected[brfss_df_selected._CHOLCHK != 9]
    brfss_df_selected._CHOLCHK.unique()
    
    #4 _BMI5 (no changes, just note that these are BMI * 100. So for example a BMI of 4018 is really 40.18)
    brfss_df_selected['_BMI5'] = brfss_df_selected['_BMI5'].div(100).round(0)
    brfss_df_selected._BMI5.unique()
    
    #5 SMOKE100
    # Change 2 to 0 because it is No
    # Remove all 7 (dont knows)
    # Remove all 9 (refused)
    brfss_df_selected['SMOKE100'] = brfss_df_selected['SMOKE100'].replace({2:0})
    brfss_df_selected = brfss_df_selected[brfss_df_selected.SMOKE100 != 7]
    brfss_df_selected = brfss_df_selected[brfss_df_selected.SMOKE100 != 9]
    brfss_df_selected.SMOKE100.unique()
    
    #6 CVDSTRK3
    # Change 2 to 0 because it is No
    # Remove all 7 (dont knows)
    # Remove all 9 (refused)
    brfss_df_selected['CVDSTRK3'] = brfss_df_selected['CVDSTRK3'].replace({2:0})
    brfss_df_selected = brfss_df_selected[brfss_df_selected.CVDSTRK3 != 7]
    brfss_df_selected = brfss_df_selected[brfss_df_selected.CVDSTRK3 != 9]
    brfss_df_selected.CVDSTRK3.unique()
    
    #7 _MICHD
    #Change 2 to 0 because this means did not have MI or CHD
    brfss_df_selected['_MICHD'] = brfss_df_selected['_MICHD'].replace({2: 0})
    brfss_df_selected._MICHD.unique()
    
    #8 _TOTINDA
    # 1 for physical activity
    # change 2 to 0 for no physical activity
    # Remove all 9 (don't know/refused)
    brfss_df_selected['_TOTINDA'] = brfss_df_selected['_TOTINDA'].replace({2:0})
    brfss_df_selected = brfss_df_selected[brfss_df_selected._TOTINDA != 9]
    brfss_df_selected._TOTINDA.unique()
    
    #9 _FRTLT1
    # Change 2 to 0. this means no fruit consumed per day. 1 will mean consumed 1 or more pieces of fruit per day 
    # remove all dont knows and missing 9
    brfss_df_selected['_FRTLT1'] = brfss_df_selected['_FRTLT1'].replace({2:0})
    brfss_df_selected = brfss_df_selected[brfss_df_selected._FRTLT1 != 9]
    brfss_df_selected._FRTLT1.unique()
    
    #10 _VEGLT1
    # Change 2 to 0. this means no vegetables consumed per day. 1 will mean consumed 1 or more pieces of vegetable per day 
    # remove all dont knows and missing 9
    brfss_df_selected['_VEGLT1'] = brfss_df_selected['_VEGLT1'].replace({2:0})
    brfss_df_selected = brfss_df_selected[brfss_df_selected._VEGLT1 != 9]
    brfss_df_selected._VEGLT1.unique()
    
    #11 _RFDRHV5
    # Change 1 to 0 (1 was no for heavy drinking). change all 2 to 1 (2 was yes for heavy drinking)
    # remove all dont knows and missing 9
    brfss_df_selected['_RFDRHV5'] = brfss_df_selected['_RFDRHV5'].replace({1:0, 2:1})
    brfss_df_selected = brfss_df_selected[brfss_df_selected._RFDRHV5 != 9]
    brfss_df_selected._RFDRHV5.unique()
    
    #12 HLTHPLN1
    # 1 is yes, change 2 to 0 because it is No health care access
    # remove 7 and 9 for don't know or refused
    brfss_df_selected['HLTHPLN1'] = brfss_df_selected['HLTHPLN1'].replace({2:0})
    brfss_df_selected = brfss_df_selected[brfss_df_selected.HLTHPLN1 != 7]
    brfss_df_selected = brfss_df_selected[brfss_df_selected.HLTHPLN1 != 9]
    brfss_df_selected.HLTHPLN1.unique()
    
    #13 MEDCOST
    # Change 2 to 0 for no, 1 is already yes
    # remove 7 for don/t know and 9 for refused
    brfss_df_selected['MEDCOST'] = brfss_df_selected['MEDCOST'].replace({2:0})
    brfss_df_selected = brfss_df_selected[brfss_df_selected.MEDCOST != 7]
    brfss_df_selected = brfss_df_selected[brfss_df_selected.MEDCOST != 9]
    brfss_df_selected.MEDCOST.unique()
    
    #14 GENHLTH
    # This is an ordinal variable that I want to keep (1 is Excellent -> 5 is Poor)
    # Remove 7 and 9 for don't know and refused
    brfss_df_selected = brfss_df_selected[brfss_df_selected.GENHLTH != 7]
    brfss_df_selected = brfss_df_selected[brfss_df_selected.GENHLTH != 9]
    brfss_df_selected.GENHLTH.unique()
    
    #15 MENTHLTH
    # already in days so keep that, scale will be 0-30
    # change 88 to 0 because it means none (no bad mental health days)
    # remove 77 and 99 for don't know not sure and refused
    brfss_df_selected['MENTHLTH'] = brfss_df_selected['MENTHLTH'].replace({88:0})
    brfss_df_selected = brfss_df_selected[brfss_df_selected.MENTHLTH != 77]
    brfss_df_selected = brfss_df_selected[brfss_df_selected.MENTHLTH != 99]
    brfss_df_selected.MENTHLTH.unique()
    
    #16 PHYSHLTH
    # already in days so keep that, scale will be 0-30
    # change 88 to 0 because it means none (no bad mental health days)
    # remove 77 and 99 for don't know not sure and refused
    brfss_df_selected['PHYSHLTH'] = brfss_df_selected['PHYSHLTH'].replace({88:0})
    brfss_df_selected = brfss_df_selected[brfss_df_selected.PHYSHLTH != 77]
    brfss_df_selected = brfss_df_selected[brfss_df_selected.PHYSHLTH != 99]
    brfss_df_selected.PHYSHLTH.unique()
    
    #17 DIFFWALK
    # change 2 to 0 for no. 1 is already yes
    # remove 7 and 9 for don't know not sure and refused
    brfss_df_selected['DIFFWALK'] = brfss_df_selected['DIFFWALK'].replace({2:0})
    brfss_df_selected = brfss_df_selected[brfss_df_selected.DIFFWALK != 7]
    brfss_df_selected = brfss_df_selected[brfss_df_selected.DIFFWALK != 9]
    brfss_df_selected.DIFFWALK.unique()
    
    #18 SEX
    # in other words - is respondent male (somewhat arbitrarily chose this change because men are at higher risk for heart disease)
    # change 2 to 0 (female as 0). Male is 1
    brfss_df_selected['SEX'] = brfss_df_selected['SEX'].replace({2:0})
    brfss_df_selected.SEX.unique()
    
    #19 _AGEG5YR
    # already ordinal. 1 is 18-24 all the way up to 13 wis 80 and older. 5 year increments.
    # remove 14 because it is don't know or missing
    brfss_df_selected = brfss_df_selected[brfss_df_selected._AGEG5YR != 14]
    brfss_df_selected._AGEG5YR.unique()
    
    #20 EDUCA
    # This is already an ordinal variable with 1 being never attended school or kindergarten only up to 6 being college 4 years or more
    # Scale here is 1-6
    # Remove 9 for refused:
    brfss_df_selected = brfss_df_selected[brfss_df_selected.EDUCA != 9]
    brfss_df_selected.EDUCA.unique()
    
    #21 INCOME2
    # Variable is already ordinal with 1 being less than $10,000 all the way up to 8 being $75,000 or more
    # Remove 77 and 99 for don't know and refused
    brfss_df_selected = brfss_df_selected[brfss_df_selected.INCOME2 != 77]
    brfss_df_selected = brfss_df_selected[brfss_df_selected.INCOME2 != 99]
    brfss_df_selected.INCOME2.unique()
    
    #Check the shape of the dataset now: We have 253,680 cleaned rows and 22 columns (1 of which is our dependent variable)
    brfss_df_selected.shape
    
    #Rename the columns to make them more readable
    brfss = brfss_df_selected.rename(columns = {'DIABETE3':'Diabetes_012', 
                                         '_RFHYPE5':'HighBP',  
                                         'TOLDHI2':'HighChol', '_CHOLCHK':'CholCheck', 
                                         '_BMI5':'BMI', 
                                         'SMOKE100':'Smoker', 
                                         'CVDSTRK3':'Stroke', '_MICHD':'HeartDiseaseorAttack', 
                                         '_TOTINDA':'PhysActivity', 
                                         '_FRTLT1':'Fruits', '_VEGLT1':"Veggies", 
                                         '_RFDRHV5':'HvyAlcoholConsump', 
                                         'HLTHPLN1':'AnyHealthcare', 'MEDCOST':'NoDocbcCost', 
                                         'GENHLTH':'GenHlth', 'MENTHLTH':'MentHlth', 'PHYSHLTH':'PhysHlth', 'DIFFWALK':'DiffWalk', 
                                         'SEX':'Sex', '_AGEG5YR':'Age', 'EDUCA':'Education', 'INCOME2':'Income' })
    
    #************************************************************************************************
    brfss.to_csv('diabetes_012_health_indicators_BRFSS2015.csv', sep=",", index=False)
    #************************************************************************************************
    
    #Copy old table to new one.
    brfss_binary = brfss
    #Change the diabetics 2 to a 1 and pre-diabetics 1 to a 0, so that we have 0 meaning non-diabetic and pre-diabetic and 1 meaning diabetic.
    brfss_binary['Diabetes_012'] = brfss_binary['Diabetes_012'].replace({1:0})
    brfss_binary['Diabetes_012'] = brfss_binary['Diabetes_012'].replace({2:1})

    #Change the column name to Diabetes_binary
    brfss_binary = brfss_binary.rename(columns = {'Diabetes_012': 'Diabetes_binary'})
    brfss_binary.Diabetes_binary.unique()
    
    #Separate the 0(No Diabetes) and 1&2(Pre-diabetes and Diabetes)
    #Get the 1s
    is1 = brfss_binary['Diabetes_binary'] == 1
    brfss_5050_1 = brfss_binary[is1]

    #Get the 0s
    is0 = brfss_binary['Diabetes_binary'] == 0
    brfss_5050_0 = brfss_binary[is0] 

    #Select the 39977 random cases from the 0 (non-diabetes group). we already have 35346 cases from the diabetes risk group
    brfss_5050_0_rand1 = brfss_5050_0.take(np.random.permutation(len(brfss_5050_0))[:35346])

    #Append the 39977 1s to the 39977 randomly selected 0s
    brfss_5050 = brfss_5050_0_rand1.append(brfss_5050_1, ignore_index = True)
    
    #Check that it worked. Now we have a dataset of 79,954 rows that is equally balanced with 50% 1 and 50% 0 for the target variable Diabetes_binary
    brfss_5050.head()
    brfss_5050.tail()
    #See the classes are perfectly balanced now
    brfss_5050.groupby(['Diabetes_binary']).size()
    print(f'brfss_5050={brfss_5050.shape}',f'brfss_binary={brfss_binary.shape}')
    #Save the 50-50 balanced dataset to csv
    #************************************************************************************************
    brfss_5050.to_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv', sep=",", index=False)
    #************************************************************************************************

    #Also save the original binary dataset to csv
    #************************************************************************************************
    brfss_binary.to_csv('diabetes_binary_health_indicators_BRFSS2015.csv', sep=",", index=False)
    #************************************************************************************************