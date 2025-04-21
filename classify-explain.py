#!/usr/bin/env python3

import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import dtreeviz
from sklearn.preprocessing import LabelEncoder
from pyxai import Learning, Explainer,Tools
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import dice_ml
from dice_ml.utils import helpers  # helper functions

import matplotlib.pyplot as plt
import shap
import imodels
from imodels import SLIMClassifier, OneRClassifier, BoostedRulesClassifier, FIGSClassifier, HSTreeClassifierCV
from imodels import SLIMRegressor, BayesianRuleListClassifier, RuleFitRegressor, GreedyRuleListClassifier
import numpy as np

def is_number(n):
	try:
		float(n) 
	except ValueError:
		return False
	return True


def parse_arguments():
	parser = argparse.ArgumentParser(description="SSPMDP-Classifier")
	parser.add_argument('--input', '-i', help='Folder training files path', required=True)

	return parser.parse_args()

def main():
   	args = parse_arguments()
	folder_path = args[0]
	labels = []
	data: dict[str, str] = {}
	with open(os.path.join(folder_path, 'trainingData.csv')) as file:
	    labels = file.readline().strip().split(',')
	    labels.append('Best Solver')
	locs = [ x for x in os.listdir(folder_path) if x in ['barabasi.csv','kronecker.csv','erdosNM.csv','smallworld.csv' ]]
	for loc in locs:
	    with open(os.path.join(folder_path, loc), 'r') as file:
		file.readline()
		for mdp in file.readlines():
		    name = mdp.split('\t')[0]
		    times = [int(x.strip()) for x in mdp.split('\t')[1:]]
		    data[name] = times
		    data[name].append(times.index(min(times)))
	#Loading des fichiers de modeles pre-existants
	locs = [ x for x in os.listdir(folder_path) if x in ['sap.csv','layered.csv','wetfloor.csv' ]]
	for loc in locs:
	    with open(os.path.join(folder_path, loc), 'r') as file:
		file.readline()
		for mdp in file.readlines():
		    name = mdp.split('\t')[0]
		    times = [int(x.strip()) for x in mdp.split('\t')[1:]]
		    data[name] = times
		    data[name].append(times.index(min(times)))
	print(len(data))

	#Loading des fichiers de modeles pre-existants
	locs = [ x for x in os.listdir(folder_path) if x in ['sapInfos.csv','layeredInfos.csv','layeredInfo2.csv','wetfloorInfos.csv' ]]
	for loc in locs:
	    with open(os.path.join(folder_path, loc), 'r') as file:
		for mdp in file.readlines():
		    name = mdp.strip().split(',')[0]
		    typeName = loc
		    typeName = typeName[:-9]
		    if name in data:
		        dat = [float(x) if is_number(x) else x for x in mdp.strip().split(',')[1:]]
		        dat.insert(0,typeName)
		        data[name] =dat + data[name]
	with open(os.path.join(folder_path, 'synthInfos.csv'), 'r') as file:
	    for mdp in file.readlines():
		name = mdp.strip().split(',')[0]
		if name in data:
		    dat = [float(x) if is_number(x) else x for x in mdp.strip().split(',')[1:]]
		    data[name] =dat + data[name]            
	with open(os.path.join(folder_path, 'otherSynth.csv'), 'r') as file:
	    for mdp in file.readlines():
		name = mdp.strip().split(',')[0]
		if name in data:
	    #We transform the current arrays to the correct dataframeFormats
		    class_names = ['Category','Nodes','Goals Ratio','SCC','Largest SCC','Clustering','Goals excentricity','avgNumActions','avgNumEffects']
		    dat = [float(x) if is_number(x) else x for x in mdp.strip().split(',')[1:]]
		    data[name] =dat + data[name]            
	brokenNames = []
	for name in data:
	    if len(data[name]) < 7:
		brokenNames.append(name)
	for n in brokenNames:
	    data.pop(n)

	    
	    #We transform the current arrays to the correct dataframeFormats
	class_names = ['Nodes','Goals Ratio','SCC','Largest SCC','Clustering','Goals excentricity','avgNumActions','avgNumEffects']
	target = "Best Solver"

	compl = ['Nodes','Goals Ratio','SCC','Largest SCC','Clustering','Goals excentricity','avgNumActions','avgNumEffects',target]

	solver_names=['VI','LRTDP','ILAOstar','TVI']
	allData = data.values()
	print(len(allData))
	di = {key:list(value) for key,value in zip(labels,zip(*allData))}
	df = pd.DataFrame(di)
	df['Goals Ratio'] = df['Goals Ratio'] / df['Nodes']
	df['Goals Ratio'] = df['Goals Ratio'] *100



	le = LabelEncoder()
	brokens = []
	for index, row in df.iterrows():
	    if row['Category'] not in ['erdosNP','erdosNM','smallWorld','barabasi','kronecker','wetfloor','sap','layered']:
		brokens.append(index)
	df.drop(brokens,inplace=True)

	print("ErdosNP solver {}".format(df['Category'].value_counts().get('erdosNP',0)))
	print("ErdosNM solver {}".format(df['Category'].value_counts().get('erdosNM',0)))
	print("smallWorld solver {}".format(df['Category'].value_counts().get('smallWorld',0)))
	print("barabasi solver {}".format(df['Category'].value_counts().get('barabasi',0)))
	print("kronecker solver {}".format(df['Category'].value_counts().get('kronecker',0)))
	print("wetfloor solver {}".format(df['Category'].value_counts().get('wetfloor',0)))
	print("sap solver {}".format(df['Category'].value_counts().get('sap',0)))
	print("layered solver {}".format(df['Category'].value_counts().get('layered',0)))

	le.fit(['erdosNP','erdosNM','smallWorld','barabasi','kronecker','wetfloor','sap','layered'])
	df['Category'] = le.transform(df['Category'])
	    #We do the main classification
	print(df)
	for col in df:
	    df[col] = df[col].replace('', '0').astype(float)
	print("First solver {}".format(df['Best Solver'].value_counts().get(0,0)))
	print("Second solver {}".format(df['Best Solver'].value_counts().get(1,0)))
	print("Third solver {}".format(df['Best Solver'].value_counts().get(2,0)))
	print("Fourth solver {}".format(df['Best Solver'].value_counts().get(3,0)))
	X = df[class_names]
	Z = df[class_names + [target]]
	#This one is the Dice code
	d = dice_ml.Data(dataframe=Z, continuous_features=['Clustering'], outcome_name=target)
	    # Using sklearn backend
	m = dice_ml.Model(model=mod, backend="sklearn")
	    # Using method=random for generating CFs
	exp = dice_ml.Dice(d, m, method="random")

	print(X_train)
	e1 = exp.generate_counterfactuals(X_train[0:300], total_CFs=10, desired_class=1,features_to_vary="all")
	e1.visualize_as_dataframe(show_only_changes=True)
	test = [x for x in e1.cf_examples_list if not hasattr(x, 'final_cfs_df') or x.final_cfs_df is not None]
	print(len(e1.cf_examples_list[0].final_cfs_df))

	imp = exp.local_feature_importance(X_train[0:300],cf_examples_list=test)
	struct = {x:0 for x in class_names}
	for elem in imp.local_importance:
	    for key in elem.keys():
		struct[key] = struct[key] + elem[key]
	for key in struct.keys():
		struct[key] = struct[key] / len(imp.local_importance)
	print(imp.local_importance)

	json_str = e1.to_json()
	with open('dice-class-1.json','w') as f:
	    json.dump(json_str,f)
	with open('dice-class-1-summary.json','w') as f:
	    json.dump(struct, f)


            
            

if __name__ == '__main__':
    main()
