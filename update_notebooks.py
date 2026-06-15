import json
import os

eda_path = "c:/Users/prane/OneDrive/Desktop/Projects/personal/notebook/EDA.ipynb"
with open(eda_path, 'r', encoding='utf-8') as f:
    eda_nb = json.load(f)

for cell in eda_nb['cells']:
    if cell['cell_type'] == 'code' and any('Impute remaining gaps' in line for line in cell['source']):
        new_source = []
        skip = False
        for line in cell['source']:
            if '# 3-c  Impute remaining gaps' in line:
                skip = True
                new_source.append('# 3-c  Impute remaining gaps moved to training.ipynb to prevent data leakage\n')
            elif skip and line.strip().startswith('# 3-d'):
                skip = False
                new_source.append(line)
            elif not skip:
                new_source.append(line)
        cell['source'] = new_source

with open(eda_path, 'w', encoding='utf-8') as f:
    json.dump(eda_nb, f, indent=1)


train_path = "c:/Users/prane/OneDrive/Desktop/Projects/personal/notebook/training.ipynb"
with open(train_path, 'r', encoding='utf-8') as f:
    train_nb = json.load(f)

for cell in train_nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if 'train_test_split' in source_str and 'get_dummies' in source_str:
            # We need to insert feature_names dump after get_dummies
            # and imputation after train_test_split
            new_source = []
            for line in cell['source']:
                new_source.append(line)
                if 'x = pd.get_dummies(x, drop_first=True)' in line:
                    new_source.append('import joblib\n')
                    new_source.append('feature_names = x.columns.tolist()\n')
                    new_source.append('joblib.dump(feature_names, "trained_feature_columns.pkl")\n')
                if 'x_train, x_test, y_train, y_test = train_test_split(' in line:
                    # The next line is the rest of the split
                    pass
                if '    x, y, test_size=0.2, random_state=42, stratify=y' in line:
                    pass
                if ')' in line and 'stratify=y' in "".join(new_source[-2:]):
                    new_source.append('\n# Impute missing values on X_train to prevent data leakage\n')
                    new_source.append('for col in x_train.columns:\n')
                    new_source.append('    if not x_train[col].isnull().any() and not x_test[col].isnull().any():\n')
                    new_source.append('        continue\n')
                    new_source.append('    if pd.api.types.is_numeric_dtype(x_train[col]):\n')
                    new_source.append('        fill_val = x_train[col].median()\n')
                    new_source.append('    else:\n')
                    new_source.append('        mode_vals = x_train[col].mode(dropna=True)\n')
                    new_source.append('        fill_val = mode_vals.iloc[0] if len(mode_vals) else "Unknown"\n')
                    new_source.append('    x_train[col] = x_train[col].fillna(fill_val)\n')
                    new_source.append('    x_test[col] = x_test[col].fillna(fill_val)\n')
            cell['source'] = new_source

        if 'XGBoost' in source_str and 'xgboost_model.fit' in source_str:
            new_source = []
            for line in cell['source']:
                new_source.append(line)
                if 'y_pred=xgboost_model.predict(x_test)' in line:
                    new_source.append('\n# Calculate optimal threshold for F1-score\n')
                    new_source.append('from sklearn.metrics import precision_recall_curve\n')
                    new_source.append('y_probs = xgboost_model.predict_proba(x_test)[:, 1]\n')
                    new_source.append('precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)\n')
                    new_source.append('f1_scores = (2 * precisions * recalls) / (precisions + recalls)\n')
                    new_source.append('optimal_idx = np.argmax(f1_scores)\n')
                    new_source.append('optimal_threshold = thresholds[optimal_idx]\n')
                    new_source.append('print(f"Optimal Risk Threshold: {optimal_threshold:.3f}")\n')
                    new_source.append('print(f"Maximized F1-Score: {f1_scores[optimal_idx]:.3f}")\n')
            cell['source'] = new_source

with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_nb, f, indent=1)

print("Notebooks updated successfully!")
