#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    # Define column structure - maintains order
    solver_columns = ['Name', 'VI', 'LRTDP', 'ILAOstar', 'TVI']
    feature_columns = ['Model', 'Nodes', 'Goals', 'SCC', 'Largest SCC',
                       'Clustering', 'Goals excentricity',
                       'avgNumActions', 'avgNumEffects']
    # Include both Name and Model in final columns
    all_columns = ['Name', 'Model'] + solver_columns[1:] + feature_columns[1:]

    data = {}

    # Load runtime CSVs first
    runtime_files = ['barabasi.csv', 'kronecker.csv', 'erdosNM.csv', 'smallworld.csv',
                     'sap.csv', 'layered.csv', 'wetfloor.csv']
    for loc in runtime_files:
        if loc in os.listdir(folder_path):
            print(f"Loading {loc}...", file=os.sys.stderr)
            with open(os.path.join(folder_path, loc), 'r') as f:
                f.readline()  # skip header
                for line in f:
                    parts = line.strip().split('\t')
                    name = parts[0]
                    times = [int(x.strip()) for x in parts[1:]]
                    data[name] = {col: val for col, val in zip(solver_columns[1:], times)}
                    data[name]['Name'] = name

    # Load and merge feature data
    infos_files = [
        ('sapInfos.csv', 'sap', {'avgNumActions': 2, 'avgNumEffects': 3}),
        ('layeredInfos.csv', 'layered', None),  # values from filename
        ('wetfloorInfos.csv', 'wetfloor', {'avgNumActions': 4}),
        ('synthInfos.csv', None, None)
    ]

    for fname, default_model, fixed_values in infos_files:
        if fname in os.listdir(folder_path):
            print(f"Loading {fname}...", file=os.sys.stderr)
            with open(os.path.join(folder_path, fname), 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    name = parts[0]
                    if name not in data:  # skip if no runtime data
                        continue

                    # Get base feature values
                    if fname == 'synthInfos.csv':
                        model = parts[1]
                        features = [safe_float(x) for x in parts[2:]]
                    else:
                        model = default_model
                        features = [safe_float(x) for x in parts[1:]]

                    # Apply model-specific adjustments
                    if fixed_values:
                        for k, v in fixed_values.items():
                            idx = feature_columns.index(k) - 1  # -1 for Model
                            features[idx] = v
                    elif default_model == 'layered':
                        # Get avgNumActions and avgNumEffects from filename
                        fn_fields = name.replace('.mdp', '').split('-')
                        avg_actions = int(fn_fields[-2])
                        max_effects = int(fn_fields[-1])
                        avg_effects = (0 + max_effects) / 2
                        features[-2] = avg_actions
                        features[-1] = avg_effects
                    elif default_model == 'wetfloor':
                        # Divide avgNumEffects by avgNumActions
                        features[-1] = features[-1] / 4.0

                    # Store all features
                    data[name]['Model'] = model
                    for col, val in zip(feature_columns[1:], features):
                        data[name][col] = val

    # Convert dict of dicts to DataFrame-ready format
    matrix_data = []
    for name in data:
        row = [data[name].get(col, 0) for col in all_columns]
        matrix_data.append(row)

    return matrix_data, all_columns


def build_dataframe(matrix_data, headers):
    # Data is already in matrix form, just convert to dataframe
    df = pd.DataFrame(matrix_data, columns=headers)

    # Ensure robust handling of columns
    if 'Goals' in df.columns and 'Nodes' in df.columns:
        df['Goals Ratio'] = df['Goals'] / df['Nodes'] * 100

    # Convert Model column to numeric Category
    le = LabelEncoder()
    mask = df['Model'].isin(CATEGORY_LIST)
    df = df[mask]
    le.fit(CATEGORY_LIST)
    df['Category'] = le.transform(df['Model'])

    df = df.fillna(0)
    # Convert numeric columns to float, leave string columns as is
    for col in df.columns:
        if col not in ['Name', 'Model']:  # Skip string columns
            df[col] = df[col].replace(' ', '0').astype(float)
    df.columns = df.columns.str.strip()
    print('Dataframe columns:', list(df.columns))

    # Compute 'Best Solver' from solver runtime columns
    solver_names = ['VI', 'LRTDP', 'ILAOstar', 'TVI']
    df['Best Solver'] = df[solver_names].idxmin(axis=1)
    return df


def classification(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (precision_recall_fscore_support,
                                 confusion_matrix)

    solver_names = ['VI', 'LRTDP', 'ILAO*', 'TVI']

    class_names = CLASS_NAMES
    target = TARGET_LABEL
    full_set = class_names + [target]
    Z = df[full_set]

    X = Z[class_names]
    y = Z[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                        test_size=0.2,
                                                        random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred,
                                                               average=None)
    accuracy = clf.score(X_test, y_test)

    cm = confusion_matrix(y_test, y_pred)

    # Feature importance
    feature_importance = dict(zip(class_names, clf.feature_importances_))

    # Format results to match paper tables
    metrics = {
        'accuracy': accuracy,
        'precision': dict(zip(solver_names, precision)),
        'recall': dict(zip(solver_names, recall)),
        'f1': dict(zip(solver_names, f1)),
        'feature_importance': feature_importance,
        'confusion_matrix': cm.tolist()
    }

    return metrics


def main():
    import numpy as np

    args = parse_arguments()
    data, headers = load_data(args.input)
    df = build_dataframe(data, headers)
    results = classification(df)

    if args.print == 'debug':
        print('Num instances:', len(data))
        print('\nDataset Statistics:')
        print(df.describe())

    elif args.print == 'df':
        print(df)

    else:  # Print all metrics by default
        print('\nClassification Results:')
        print(f"Global Accuracy: {results['accuracy'] * 100:.2f}%")

        print('\nPer-Solver Metrics:')
        solver_names = ['VI', 'LRTDP', 'ILAO*', 'TVI']
        print(f"{'Solver':10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print('-' * 45)
        for solver in solver_names:
            print(f"{solver:10} {results['precision'][solver] * 100:>10.2f}% {results['recall'][solver] * 100:>10.2f}% {results['f1'][solver]:>10.3f}")

        print('\nFeature Importance:')
        sorted_features = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features:
            print(f"{feature:20} {importance:.3f}")

        print('\nConfusion Matrix:')
        cm = np.array(results['confusion_matrix'])
        print('True \\ Pred'.ljust(12) + ' '.join(f'{s:>8}' for s in solver_names))
        print('-' * 50)
        for i, solver in enumerate(solver_names):
            print(f'{solver:<12}' + ' '.join(f'{n:>8}' for n in cm[i]))


if __name__ == '__main__':
    main()
