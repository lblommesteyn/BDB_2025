import pandas as pd
import numpy as np
import argparse
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Features to use for prediction
FEATURE_COLUMNS = [
    'dacs_final',
    'peak_collapse_rate',
    'time_to_50pct',
    'ce_final_norm',
    'bfoi',
    'coverage_intensity',
    'pursuit_eff_mean',
    'hrt_mean',
    'arrival_mean'
]

def train_model(summary_path: str, out_dir: str):
    print(f"Loading summary from {summary_path}...")
    df = pd.read_parquet(summary_path)
    
    # Filter for valid pass results
    # Mapping based on inspection (adjust if needed)
    # Assuming 'C'/'Complete', 'I'/'Incomplete', 'IN'/'Interception'
    # We will standardize to 'catch', 'incomplete', 'interception'
    
    # Normalize pass_result
    df['pass_result'] = df['pass_result'].astype(str).str.strip()
    
    # Define mapping
    mapping = {
        'C': 'catch', 'Complete': 'catch',
        'I': 'incomplete', 'Incomplete': 'incomplete',
        'IN': 'interception', 'Interception': 'interception',
        'S': 'sack', 'Sack': 'sack', # Sacks might be filtered out
        'R': 'run', 'Run': 'run' # Runs should be filtered out
    }
    
    df['outcome'] = df['pass_result'].map(mapping)
    
    # Filter for target classes
    target_classes = ['catch', 'incomplete', 'interception']
    df_model = df[df['outcome'].isin(target_classes)].copy()
    
    print(f"Filtered to {len(df_model)} plays with valid outcomes: {df_model['outcome'].value_counts().to_dict()}")
    
    if len(df_model) < 50:
        print("Warning: Too few samples to train a reliable model.")
    
    # Prepare features and target
    X = df_model[FEATURE_COLUMNS].fillna(0.0) # Handle NaNs (e.g. arrival_mean might be NaN if no arrival)
    y = df_model['outcome']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to benchmark
    models = {
        'LogisticRegression': LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
    }
    
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric='mlogloss', use_label_encoder=False)
        print("XGBoost included in benchmark.")
    except ImportError:
        print("XGBoost not found, skipping.")

    best_name = None
    best_score = -1.0
    best_model = None
    best_report = ""
    
    results = []

    print(f"\nBenchmarking {len(models)} models...")
    
    for name, clf in models.items():
        print(f"Training {name}...")
        
        # XGBoost requires encoded labels (0, 1, 2)
        if name == 'XGBoost':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc = le.transform(y_test)
            clf.fit(X_train_scaled, y_train_enc)
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)
            # Decode for report if needed, but classification_report handles it if we pass names
            # Actually y_test is strings, y_pred is ints. Need to decode y_pred or encode y_test.
            # Let's keep y_test as is and decode y_pred for consistency
            y_pred = le.inverse_transform(y_pred)
        else:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)

        report = classification_report(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        except:
            auc = 0.0
            
        print(f"  {name} AUC: {auc:.4f}")
        results.append(f"{name}: AUC={auc:.4f}")
        
        if auc > best_score:
            best_score = auc
            best_name = name
            best_model = clf
            best_report = report

    print(f"\nBest Model: {best_name} (AUC={best_score:.4f})")
    print("Best Classification Report:")
    print(best_report)
    
    with open('model_metrics.txt', 'w') as f:
        f.write(f"Best Model: {best_name}\n")
        f.write(f"AUC: {best_score:.4f}\n\n")
        f.write("Benchmark Results:\n")
        f.write("\n".join(results) + "\n\n")
        f.write(best_report)

    # Save best model
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'outcome_model.joblib')
    
    bundle = {
        'clf': best_model,
        'scaler': scaler,
        'features': FEATURE_COLUMNS,
        'classes': best_model.classes_ if best_name != 'XGBoost' else ['catch', 'incomplete', 'interception'] # XGB classes might be ints
    }
    
    # Fix classes for XGBoost if needed
    if best_name == 'XGBoost':
         # We need to ensure classes_ maps 0,1,2 to the string labels correctly
         # In the loop we used LabelEncoder. The order is sorted(unique(y)).
         # 'catch', 'incomplete', 'interception' -> sorted: 'catch', 'incomplete', 'interception'
         # So 0=catch, 1=incomplete, 2=interception.
         bundle['classes'] = np.array(['catch', 'incomplete', 'interception'])

    joblib.dump(bundle, model_path)
    print(f"\nBest model saved to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True, help='Path to season_summary.parquet')
    parser.add_argument('--out', required=True, help='Output directory for model')
    args = parser.parse_args()
    
    train_model(args.summary, args.out)
