"""Compare fixed tree models with notebook 02 on the same selection payments."""
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from run_baselines import ROOT, action_costs, action_profits, cancel_from_probability, selection_metrics, setup_tracking, sha256


def main():
    baseline = json.loads((ROOT / 'reports/baseline-results.json').read_text())
    audit = json.loads((ROOT / 'reports/duplicate-split-audit.json').read_text())
    feature_report_path = ROOT / 'reports/feature-engineering-results.json'
    feature_report = json.loads(feature_report_path.read_text())
    if feature_report['metadata']['baseline_report_sha256'] != sha256(ROOT / 'reports/baseline-results.json'):
        raise ValueError('Stage 03 feature comparison is stale; rerun it before experiments')
    costs = json.loads((ROOT / 'config/cost-scenario.json').read_text())
    protocol = json.loads((ROOT / 'config/evaluation.json').read_text())
    assignment_path = ROOT / audit['assignments']['path']
    for path, expected in [(ROOT / 'config/evaluation.json', audit['config_sha256']),
                           (assignment_path, audit['assignments']['sha256']),
                           (ROOT / 'config/evaluation.json', baseline['metadata']['protocol_sha256']),
                           (ROOT / 'config/cost-scenario.json', baseline['metadata']['cost_config_sha256']),
                           (ROOT / 'poetry.lock', baseline['metadata']['poetry_lock_sha256'])]:
        if sha256(path) != expected:
            raise ValueError(f'Stale split or baseline lineage: {path}; rerun audit/baselines')
    if baseline['metadata']['assignments_sha256'] != audit['assignments']['sha256']:
        raise ValueError('Baseline and experiment assignments differ')
    assignments = pd.read_csv(assignment_path).set_index('source_row_id')
    if not audit['group_disjoint'] or assignments.groupby('pattern_group_id')['split'].nunique().max() != 1:
        raise ValueError('Patterns overlap across splits')
    parts, offset = [], 0
    for source in audit['source_files']:
        path = ROOT / source['path']
        if sha256(path) != source['sha256']:
            raise ValueError(f'Source changed: {path}')
        for chunk in pd.read_csv(path, chunksize=50000):
            chunk.index = np.arange(offset, offset + len(chunk))
            offset += len(chunk)
            parts.append(chunk.loc[assignments.loc[chunk.index, 'split'].isin(['train', 'selection']).to_numpy()])
    data = pd.concat(parts)
    split = assignments.loc[data.index, 'split']
    train, selection = data.loc[split.eq('train')], data.loc[split.eq('selection')]
    assert len(train) == baseline['metadata']['train_rows'] and int(train.Class.sum()) == baseline['metadata']['train_frauds']
    assert len(selection) == baseline['metadata']['selection_rows'] and int(selection.Class.sum()) == baseline['metadata']['selection_frauds']
    columns = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
    baseline_predictions = pd.read_csv(ROOT / baseline['selection_predictions']['path'])
    assert sha256(ROOT / baseline['selection_predictions']['path']) == baseline['selection_predictions']['sha256']
    assert set(selection.index) == set(baseline_predictions.source_row_id)
    assert set(baseline_predictions.model) == {x['name'] for x in baseline['results']}
    setup_tracking('04-fixed-model-comparison')
    metadata = {
        'evaluated_split': 'selection', 'train_rows': len(train), 'train_frauds': int(train.Class.sum()),
        'selection_rows': len(selection), 'selection_frauds': int(selection.Class.sum()),
        'evaluation_split_scored': False, 'calibration_split_used': False,
        'protocol': protocol['version'], 'random_seed': protocol['random_seed'],
        'code_sha256': sha256(Path(__file__)), 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'protocol_sha256': sha256(ROOT / 'config/evaluation.json'),
        'cost_config_sha256': sha256(ROOT / 'config/cost-scenario.json'),
        'assignments_sha256': sha256(assignment_path), 'poetry_lock_sha256': sha256(ROOT / 'poetry.lock'),
        'baseline_report_sha256': sha256(ROOT / 'reports/baseline-results.json'),
        'feature_report_sha256': sha256(feature_report_path),
        'feature_decision': 'retain_original_30_inputs',
        'source_files': audit['source_files'],
        'versions': {name: version(name) for name in ['scikit-learn', 'lightgbm', 'mlflow']},
        'notes': ['Fixed settings only; no tuning or post-hoc calibration.',
                  'Probability-policy profits are provisional until calibration.',
                  'Selection metrics describe this development population, not independent final evaluation.'],
    }
    models = {
        'random_forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=4),
        'lightgbm': LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=40,
                                  random_state=42, n_jobs=4, verbosity=-1),
    }
    output_dir = ROOT / 'models/experiments'
    output_dir.mkdir(parents=True, exist_ok=True)
    results, frames = [], []
    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(train[columns], train.Class)
            score = model.predict_proba(selection[columns])[:, 1]
            canceled = cancel_from_probability(score, selection.Amount, costs)
            metrics = selection_metrics(selection.Class, selection.Amount, score, canceled, costs, True)
            bundle = {'model': model, 'input_columns': columns, 'policy': 'cancel_when_expected_profit_negative',
                      'cost_scenario': costs, 'protocol': protocol, 'lineage': metadata, 'posthoc_calibrated': False}
            path = output_dir / f'{name}.joblib'
            joblib.dump(bundle, path)
            assert np.allclose(joblib.load(path)['model'].predict_proba(selection[columns].iloc[:20])[:, 1], score[:20])
            mlflow.log_artifact(str(path), artifact_path='fitted_model')
            mlflow.log_params({'model': name, 'split': 'selection', 'include_time': True, 'calibrated': False, **model.get_params()})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None and np.isfinite(v)})
            mlflow.set_tags({k: metadata[k] for k in ['code_sha256', 'protocol_sha256', 'assignments_sha256', 'cost_config_sha256']})
            mlflow.log_dict(metadata, 'lineage.json')
            results.append({'name': name, 'mlflow_run_id': run.info.run_id, 'parameters': model.get_params(), **metrics})
            frames.append(pd.DataFrame({'source_row_id': selection.index, 'model': name, 'score': score,
                                        'canceled': canceled, 'Class': selection.Class.to_numpy(), 'Amount': selection.Amount.to_numpy(),
                                        'simulated_cost': action_costs(selection.Class, selection.Amount, canceled, costs),
                                        'simulated_profit': action_profits(selection.Class, selection.Amount, canceled, costs)}))
    path = ROOT / 'data/processed/experiment-selection-predictions.csv'
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    report = {'metadata': metadata, 'baseline_results': baseline['results'], 'results': results,
              'selection_predictions': {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}}
    (ROOT / 'reports/experiment-results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    display = pd.DataFrame(baseline['results'] + results)
    print(display[['name', 'average_precision', 'brier_score', 'fraud_canceled_recall',
                   'legitimate_cancellation_rate', 'simulated_profit_per_1000']].round(6).to_string(index=False))
    print('Selection only; calibration and evaluation splits were not scored.')


if __name__ == '__main__':
    main()
