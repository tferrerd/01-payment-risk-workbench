"""Audit training features and test one prespecified elapsed-time transformation."""
import json
import subprocess
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from feature_transforms import TimeCycle
from run_baselines import ROOT, action_costs, action_profits, cancel_from_probability, selection_metrics, setup_tracking, sha256


def main():
    baseline = json.loads((ROOT / 'reports/baseline-results.json').read_text())
    audit = json.loads((ROOT / 'reports/duplicate-split-audit.json').read_text())
    costs = json.loads((ROOT / 'config/cost-scenario.json').read_text())
    for path, expected in [(ROOT / 'config/evaluation.json', audit['config_sha256']),
                           (ROOT / audit['assignments']['path'], audit['assignments']['sha256']),
                           (ROOT / 'config/evaluation.json', baseline['metadata']['protocol_sha256']),
                           (ROOT / 'config/cost-scenario.json', baseline['metadata']['cost_config_sha256'])]:
        if sha256(path) != expected:
            raise ValueError(f'Stale baseline or split: {path}')
    assignments = pd.read_csv(ROOT / audit['assignments']['path']).set_index('source_row_id')
    if not audit['group_disjoint'] or assignments.groupby('pattern_group_id').split.nunique().max() != 1:
        raise ValueError('Pattern overlap')
    chunks, offset = [], 0
    for source in audit['source_files']:
        path = ROOT / source['path']
        if sha256(path) != source['sha256']:
            raise ValueError(f'Source changed: {path}')
        for chunk in pd.read_csv(path, chunksize=50000):
            chunk.index = np.arange(offset, offset + len(chunk))
            offset += len(chunk)
            chunks.append(chunk.loc[assignments.loc[chunk.index, 'split'].isin(['train', 'selection']).to_numpy()])
    data = pd.concat(chunks)
    splits = assignments.loc[data.index, 'split']
    train, selection = data.loc[splits.eq('train')], data.loc[splits.eq('selection')]
    assert len(train) == baseline['metadata']['train_rows'] and len(selection) == baseline['metadata']['selection_rows']
    features = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
    if data[features].isna().any().any() or not np.isfinite(data[features].to_numpy()).all() or (data.Amount < 0).any():
        raise ValueError('Feature inputs must be finite, complete, and have nonnegative amounts')
    baseline_result = next(x for x in baseline['results'] if x['name'] == 'logistic_regression')
    baseline_predictions = pd.read_csv(ROOT / baseline['selection_predictions']['path'])
    if sha256(ROOT / baseline['selection_predictions']['path']) != baseline['selection_predictions']['sha256']:
        raise ValueError('Baseline predictions changed')
    assert set(baseline_predictions.source_row_id) == set(selection.index)
    transformed = ColumnTransformer([
        ('components', StandardScaler(), [f'V{i}' for i in range(1, 29)]),
        ('amount', Pipeline([('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
                            ('scale', StandardScaler())]), ['Amount']),
        ('time', StandardScaler(), ['Time']),
        ('time_cycle', TimeCycle(), ['Time']),
    ])
    model = Pipeline([('preprocessing', transformed),
                      ('classifier', LogisticRegression(C=1, solver='lbfgs', max_iter=2000, random_state=42))])
    model.fit(train[features], train.Class)
    assert model.named_steps['preprocessing'].named_transformers_['components'].n_samples_seen_ == len(train)
    score = model.predict_proba(selection[features])[:, 1]
    canceled = cancel_from_probability(score, selection.Amount, costs)
    metrics = selection_metrics(selection.Class, selection.Amount, score, canceled, costs, True)
    metadata = {'evaluated_split': 'selection', 'train_rows': len(train), 'selection_rows': len(selection),
                'evaluation_split_scored': False, 'calibration_split_used': False,
                'code_sha256': sha256(Path(__file__)), 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'assignments_sha256': audit['assignments']['sha256'], 'protocol_sha256': sha256(ROOT / 'config/evaluation.json'),
                'cost_config_sha256': sha256(ROOT / 'config/cost-scenario.json'),
                'baseline_report_sha256': sha256(ROOT / 'reports/baseline-results.json'),
                'source_files': audit['source_files'],
                'feature_hypothesis': 'Add sin/cos of Time mod 86400 to the existing logistic inputs; phase is relative to first transaction, not local clock time.'}
    path = ROOT / 'models/features/logistic_time_cycle.joblib'
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': model, 'input_columns': features, 'lineage': metadata, 'posthoc_calibrated': False}, path)
    assert np.allclose(joblib.load(path)['pipeline'].predict_proba(selection[features].iloc[:20])[:, 1], score[:20])
    setup_tracking('03-feature-engineering')
    with mlflow.start_run(run_name='logistic_time_cycle') as run:
        mlflow.log_params({'model': 'logistic_regression', 'C': 1, 'feature_addition': 'sin_cos_time_86400',
                           'split': 'selection', 'calibrated': False, 'include_time': True})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None and np.isfinite(v)})
        mlflow.set_tags({k: metadata[k] for k in ['code_sha256', 'assignments_sha256', 'protocol_sha256']})
        mlflow.log_dict(metadata, 'lineage.json')
        mlflow.log_artifact(str(path), artifact_path='fitted_pipeline')
    prediction_path = ROOT / 'data/processed/feature-selection-predictions.csv'
    pd.DataFrame({'source_row_id': selection.index, 'score': score, 'canceled': canceled,
                  'Class': selection.Class.to_numpy(), 'Amount': selection.Amount.to_numpy(),
                  'simulated_cost': action_costs(selection.Class, selection.Amount, canceled, costs),
                  'simulated_profit': action_profits(selection.Class, selection.Amount, canceled, costs)}).to_csv(prediction_path, index=False)
    report = {'metadata': metadata, 'baseline_logistic': baseline_result,
              'time_cycle_logistic': {'name': 'logistic_time_cycle', 'mlflow_run_id': run.info.run_id, **metrics},
              'selection_predictions': {'path': str(prediction_path.relative_to(ROOT)), 'sha256': sha256(prediction_path)}}
    (ROOT / 'reports/feature-engineering-results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(pd.DataFrame([baseline_result, report['time_cycle_logistic']])
          [['name', 'average_precision', 'brier_score', 'fraud_canceled_recall',
            'legitimate_cancellation_rate', 'simulated_profit_per_1000']].round(6).to_string(index=False))
    print('Selection only; calibration and evaluation splits were not scored.')


if __name__ == '__main__':
    main()
