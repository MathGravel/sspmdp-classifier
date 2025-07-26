#!/usr/bin/env python3
import os
import argparse
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import dice_ml
from dice_ml.utils import helpers

PRINT_CHOICES = ['summary', 'cf', 'importance', 'df', 'debug']
CATEGORY_LIST = [
    'erdosNP', 'erdosNM', 'smallWorld', 'barabasi',
    'kronecker', 'wetfloor', 'sap', 'layered'
]
CLASS_NAMES = [
    'Nodes', 'Goals Ratio', 'SCC', 'Largest SCC',
    'Clustering', 'Goals excentricity', 'avgNumActions', 'avgNumEffects'
]
TARGET_LABEL = 'Best Solver'


def parse_arguments():
    parser = argparse.ArgumentParser(description="SSPMDP-Classifier")
    parser.add_argument('--input', '-i', required=True,
                        help='Folder training files path')
    parser.add_argument('--print', choices=PRINT_CHOICES, default='summary',
                        help='Output to print')
    return parser.parse_args()


def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return val


def load_data(folder_path):
    labels = []
    data = {}
    with open(os.path.join(folder_path, 'trainingData.csv')) as file:
        labels = file.readline().strip().split(',')
        labels.append(TARGET_LABEL)
    csv_groups = [
        ['barabasi.csv', 'kronecker.csv', 'erdosNM.csv', 'smallworld.csv'],
        ['sap.csv', 'layered.csv', 'wetfloor.csv']
    ]
    for csv_files in csv_groups:
        for loc in csv_files:
            if loc in os.listdir(folder_path):
                with open(os.path.join(folder_path, loc), 'r') as f:
                    f.readline()
                    for mdp in f:
                        mdp_split = mdp.split('\t')
                        name = mdp_split[0]
                        times = [int(x.strip()) for x in mdp_split[1:]]
                        data[name] = times
                        data[name].append(times.index(min(times)))
    extra_files = [
        ('sapInfos.csv', True), ('layeredInfos.csv', True),
        ('layeredInfo2.csv', True), ('wetfloorInfos.csv', True),
        ('synthInfos.csv', False), ('otherSynth.csv', False)
    ]
    for fname, insert_type in extra_files:
        if fname in os.listdir(folder_path):
            with open(os.path.join(folder_path, fname), 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    name = parts[0]
                    dat = [safe_float(x) for x in parts[1:]]
                    if insert_type:
                        type_name = fname[:-9]
                        dat.insert(0, type_name)
                    if name in data:
                        data[name] = dat + data[name]
    data = {k: v for k, v in data.items() if len(v) >= 7}
    return data, labels


def build_dataframe(data, labels):
    all_data = list(data.values())
    di = {k: list(v) for k, v in zip(labels, zip(*all_data))}
    df = pd.DataFrame(di)
    df['Goals Ratio'] = df['Goals Ratio'] / df['Nodes'] * 100
    le = LabelEncoder()
    mask = df['Category'].isin(CATEGORY_LIST)
    df = df[mask]
    le.fit(CATEGORY_LIST)
    df['Category'] = le.transform(df['Category'])
    for col in df:
        df[col] = df[col].replace('', '0').astype(float)
    return df


def classification(df):
    class_names = CLASS_NAMES
    target = TARGET_LABEL
    full_set = class_names + [target]
    Z = df[full_set]
    d = dice_ml.Data(
        dataframe=Z,
        continuous_features=['Clustering'],
        outcome_name=target
    )
    mod = helpers.get_trained_model_for_dataset(
        Z, backend="sklearn"
    )
    m = dice_ml.Model(model=mod, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")
    X_train = Z[class_names]
    cf_examples = exp.generate_counterfactuals(
        X_train[:300],
        total_CFs=10,
        desired_class=1,
        features_to_vary="all"
    )
    valid_cfs = [
        x for x in cf_examples.cf_examples_list
        if hasattr(x, 'final_cfs_df') and x.final_cfs_df is not None
    ]
    imp = exp.local_feature_importance(
        X_train[:300], cf_examples_list=valid_cfs
    )
    struct = {x: 0 for x in class_names}
    for elem in imp.local_importance:
        for key in elem:
            struct[key] += elem[key]
    for key in struct:
        struct[key] /= len(imp.local_importance)
    return {
        'df': df,
        'cf': cf_examples,
        'importance': imp.local_importance,
        'summary': struct
    }


def main():
    args = parse_arguments()
    data, labels = load_data(args.input)
    df = build_dataframe(data, labels)
    results = classification(df)
    if args.print == 'debug':
        print('Num instances:', len(data))
        print(df.describe())
    elif args.print == 'df':
        print(df)
    elif args.print == 'importance':
        print(json.dumps(results['importance']))
    elif args.print == 'cf':
        print(
            results['cf'].visualize_as_dataframe(
                show_only_changes=True
            )
        )
    elif args.print == 'summary':
        print(json.dumps(results['summary'], indent=2))


if __name__ == '__main__':
    main()
