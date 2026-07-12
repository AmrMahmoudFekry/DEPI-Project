"""
SME Risk Intelligence — Full Modeling Pipeline
=============================================
Run this script to retrain on new data:
    python modeling_pipeline.py

Outputs:
    • pipeline.pkl                  — ready-to-use sklearn pipeline
    • model_comparison_results.json — dashboard comparison data
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from utils.helper import SMEFeatureEngineer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')




def load_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    return df


def build_results_json(results: dict, final_metrics: dict, best_name: str, features: list) -> dict:
    return {
        'model_comparison': results,
        'final_metrics': final_metrics,
        'best_model': best_name,
        'features_used': features,
    }


def main():
    print('=' * 60)
    print('  SME Risk Intelligence — Modeling Pipeline')
    print('=' * 60)

    root_dir = Path(__file__).resolve().parent
    data_path = root_dir / 'Data' / 'SMEs_Data.csv'

    df = load_data(data_path)
    print(f"\n[OK] Data loaded  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Low Risk (0) : {(df['risk_sharp'] == 0).sum():,}")
    print(f"  High Risk (1): {(df['risk_sharp'] == 1).sum():,}")

    TARGET = 'risk_sharp'
    FEATURES = [c for c in df.columns if c != TARGET]

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"\n[OK] Train: {len(X_train):,} | Test: {len(X_test):,}")

    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
    ])

    MODELS = {
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.15,
            max_depth=3,
            min_samples_leaf=3,
            subsample=1.0,
            random_state=42,
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            learning_rate=0.15,
            subsample=1.0,
            max_depth=5,
            colsample_bytree=1.0,
            random_state=42,
            n_jobs=-1,
        ),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.15,
            max_depth=3,
            min_samples_leaf=30,
            class_weight='balanced',
            random_state=42,
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=2,
            class_weight='balanced',
            max_features='log2',
            random_state=42,
            n_jobs=-1,
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=150, 
            learning_rate=1.0, 
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            C=0.5, 
            max_iter=500,
            class_weight='balanced',
            solver='lbfgs',
            random_state=42,
        ),
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    results = {}
    best_name = None
    best_auc = 0.0

    print('\n' + '=' * 60)
    print('  MODEL COMPARISON')
    print('=' * 60)

    for name, model in MODELS.items():
        print(f"\n[RUN] {name}...")

        pipe = Pipeline([
            ('feat_eng', SMEFeatureEngineer()),
            ('preprocessor', preprocessor),
            ('classifier', model),
        ])

        cv_aucs = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1
        )

        pipe.fit(X_train, y_train)
        train_proba = pipe.predict_proba(X_train)[:, 1]
        yp = pipe.predict(X_test)
        yproba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, yp)
        prec = precision_score(y_test, yp)
        rec = recall_score(y_test, yp)
        f1 = f1_score(y_test, yp)
        auc = roc_auc_score(y_test, yproba)
        avg_prec = average_precision_score(y_test, yproba)
        train_auc = roc_auc_score(y_train, train_proba)
        cv_score = cv_aucs.mean()

        results[name] = {
            'cv_roc_auc_mean': round(cv_score * 100, 3),
            'cv_roc_auc_std': round(cv_aucs.std() * 100, 3),
            'train_roc_auc': round(train_auc * 100, 3),
            'test_accuracy': round(acc * 100, 3),
            'test_precision': round(prec * 100, 3),
            'test_recall': round(rec * 100, 3),
            'test_f1': round(f1 * 100, 3),
            'test_roc_auc': round(auc * 100, 3),
            'test_avg_precision': round(avg_prec * 100, 3),
        }

        print(
            f"   CV  ROC-AUC : {cv_score * 100:.3f}% ± {cv_aucs.std() * 100:.3f}%"
        )
        print(
            f"   Train ROC-AUC: {train_auc * 100:.3f}% | Test ROC-AUC: {auc * 100:.3f}% | AP: {avg_prec * 100:.3f}%"
        )

        if cv_score > best_auc:
            best_auc = cv_score
            best_name = name

    print('\n' + '=' * 60)
    print('  RESULTS SUMMARY (sorted by Test ROC-AUC)')
    print('=' * 90)
    print(f"{'Model':<22} {'CV AUC':>8} {'Train AUC':>10} {'Test AUC':>9} {'AP':>7} {'ACC':>7} {'F1':>7}")
    print('-' * 90)
    for n, r in sorted(results.items(), key=lambda x: x[1]['test_roc_auc'], reverse=True):
        tag = ' [BEST]' if n == best_name else ''
        print(
            f"{n:<22} {r['cv_roc_auc_mean']:>7.3f}%  {r['train_roc_auc']:>7.3f}%  {r['test_roc_auc']:>8.3f}%  {r['test_avg_precision']:>7.3f}%  {r['test_accuracy']:>6.2f}%  {r['test_f1']:>6.2f}%{tag}"
        )

    print(f"\n[OK] Winner: {best_name} (selected by CV ROC-AUC)")
    print('  Building final calibrated pipeline...')

    final_pipeline = Pipeline([
        ('feat_eng', SMEFeatureEngineer()),
        ('preprocessor', preprocessor),
        ('classifier', CalibratedClassifierCV(
            estimator=clone(MODELS[best_name]),
            method='isotonic',
            cv=5,
        )),
    ])
    final_pipeline.fit(X_train, y_train)

    yp = final_pipeline.predict(X_test)
    yproba = final_pipeline.predict_proba(X_test)[:, 1]
 
    final_metrics = {
        'best_model': best_name,
        'test_accuracy': round(accuracy_score(y_test, yp) * 100, 2),
        'test_precision': round(precision_score(y_test, yp) * 100, 2),
        'test_recall': round(recall_score(y_test, yp) * 100, 2),
        'test_f1': round(f1_score(y_test, yp) * 100, 2),
        'test_roc_auc': round(roc_auc_score(y_test, yproba) * 100, 2),
    }

    print('\n' + '=' * 60)
    print('  FINAL CALIBRATED PIPELINE METRICS')
    print('=' * 60)
    print(f"  Best Model : {final_metrics['best_model']}")
    print(f"  Accuracy   : {final_metrics['test_accuracy']}%")
    print(f"  Precision  : {final_metrics['test_precision']}%")
    print(f"  Recall     : {final_metrics['test_recall']}%")
    print(f"  F1 Score   : {final_metrics['test_f1']}%")
    print(f"  ROC AUC    : {final_metrics['test_roc_auc']}%")
    print('\nClassification Report:')
    print(classification_report(y_test, yp))

    joblib.dump(final_pipeline, root_dir / 'pipeline.pkl')
    print('[OK] pipeline.pkl saved')

    comparison_df = pd.DataFrame.from_dict(
        results, orient='index'
    ).reset_index().rename(columns={'index': 'model'})
    comparison_df.to_csv(root_dir / 'model_comparison.csv', index=False)
    print('[OK] model_comparison.csv saved')

    with open(root_dir / 'model_comparison_results.json', 'w') as f:
        json.dump(
            build_results_json(results, final_metrics, best_name, FEATURES),
            f,
            indent=2,
        )
    print('[OK] model_comparison_results.json saved')

    print('\n' + '=' * 60)
    print('  PIPELINE COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    main()
