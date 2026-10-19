"""Calibrate fixed tree candidates on the calibration split; compare on selection."""
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from run_baselines import ROOT, action_costs, action_profits, cancel_from_probability, selection_metrics, setup_tracking, sha256


def main():
    experiment_path = ROOT / 'reports/experiment-results.json'
    experiment = json.loads(experiment_path.read_text())
    audit = json.loads((ROOT / 'reports/duplicate-split-audit.json').read_text())
    costs = json.loads((ROOT / 'config/cost-scenario.json').read_text())
    protocol = json.loads((ROOT / 'config/evaluation.json').read_text())
    assignment_path = ROOT / audit['assignments']['path']
    for path, expected in [(ROOT / 'config/evaluation.json', audit['config_sha256']),
                           (assignment_path, audit['assignments']['sha256']),
                           (ROOT / 'config/evaluation.json', experiment['metadata']['protocol_sha256']),
                           (ROOT / 'config/cost-scenario.json', experiment['metadata']['cost_config_sha256']),
                           (ROOT / 'poetry.lock', experiment['metadata']['poetry_lock_sha256'])]:
        if sha256(path) != expected:
            raise ValueError(f'Stale protocol, assignment, or experiment: {path}')
    assignments = pd.read_csv(assignment_path).set_index('source_row_id')
    if not audit['group_disjoint'] or assignments.groupby('pattern_group_id').split.nunique().max() != 1:
        raise ValueError('Pattern groups span splits')
    parts, offset = [], 0
    for source in audit['source_files']:
        path = ROOT / source['path']
        if sha256(path) != source['sha256']:
            raise ValueError(f'Source changed: {path}')
        for chunk in pd.read_csv(path, chunksize=50000):
            chunk.index = np.arange(offset, offset + len(chunk))
            offset += len(chunk)
            parts.append(chunk.loc[assignments.loc[chunk.index, 'split'].isin(['calibration', 'selection']).to_numpy()])
    data = pd.concat(parts)
    split = assignments.loc[data.index, 'split']
    calibration, selection = data.loc[split.eq('calibration')], data.loc[split.eq('selection')]
    if not data.index.is_unique or not np.isfinite(data.to_numpy()).all():
        raise ValueError('Invalid development rows')
    if len(calibration) != 28452 or int(calibration.Class.sum()) != 48:
        raise ValueError('Unexpected calibration population')
    if len(selection) != experiment['metadata']['selection_rows'] or int(selection.Class.sum()) != experiment['metadata']['selection_frauds']:
        raise ValueError('Unexpected selection population')
    if set(calibration.index) & set(selection.index):
        raise ValueError('Calibration and selection rows overlap')
    columns = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
    old_path = ROOT / experiment['selection_predictions']['path']
    if sha256(old_path) != experiment['selection_predictions']['sha256']:
        raise ValueError('Raw experiment predictions changed')
    old = pd.read_csv(old_path)
    setup_tracking('06-calibration-and-policy')
    groups = assignments.loc[calibration.index, 'pattern_group_id']
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(calibration, calibration.Class, groups))
    for fit_indices, held_indices in folds:
        if set(groups.iloc[fit_indices]) & set(groups.iloc[held_indices]):
            raise ValueError('Calibration patterns overlap across folds')
        if calibration.Class.iloc[fit_indices].nunique() != 2 or calibration.Class.iloc[held_indices].nunique() != 2:
            raise ValueError('Each calibration fold requires both classes')
    metadata = {
        'protocol': protocol['version'], 'calibration_method_selection': 'minimum grouped 3-fold calibration Brier; identity, sigmoid, temperature, isotonic',
        'calibration_rows': len(calibration), 'calibration_frauds': int(calibration.Class.sum()),
        'selection_rows': len(selection), 'selection_frauds': int(selection.Class.sum()),
        'training_split_used_for_calibrator': False, 'evaluation_split_scored': False,
        'code_sha256': sha256(Path(__file__)), 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'assignment_sha256': sha256(assignment_path), 'protocol_sha256': sha256(ROOT / 'config/evaluation.json'),
        'cost_config_sha256': sha256(ROOT / 'config/cost-scenario.json'),
        'experiment_report_sha256': sha256(experiment_path), 'poetry_lock_sha256': sha256(ROOT / 'poetry.lock'),
        'source_files': audit['source_files'],
        'versions': {name: version(name) for name in ['scikit-learn', 'lightgbm', 'mlflow']},
        'notes': ['Base classifiers were fitted on training only and frozen during post-hoc calibration.',
                  'Calibration method selected by grouped 3-fold out-of-fold Brier on calibration only; identity means no post-hoc mapping.',
                  'Full-calibration diagnostics are in-sample for fitted mappings.',
                  'Selection results reuse a set already used for model comparison and are development evidence.',
                  'The 0.20 policy threshold follows fixed assumed economics; it was not tuned.',
                  'Evaluation labels remain unused; no production impact claim.'],
    }
    results, selection_frames, calibration_frames = [], [], []
    output_dir = ROOT / 'models/calibrated'
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ['random_forest', 'lightgbm']:
        raw_bundle_path = ROOT / f'models/experiments/{name}.joblib'
        raw_bundle = joblib.load(raw_bundle_path)
        if raw_bundle['input_columns'] != columns or raw_bundle['lineage']['code_sha256'] != experiment['metadata']['code_sha256']:
            raise ValueError(f'Stale base model: {name}')
        model = raw_bundle['model']
        raw_selection = model.predict_proba(selection[columns])[:, 1]
        saved = old.loc[old.model.eq(name)].set_index('source_row_id').loc[selection.index]
        if len(saved) != len(selection) or not np.allclose(raw_selection, saved.score, atol=1e-12):
            raise ValueError(f'Raw predictions changed: {name}')
        raw_calibration = model.predict_proba(calibration[columns])[:, 1]
        method_checks = []
        for method in ['identity', 'sigmoid', 'temperature', 'isotonic']:
            if method == 'identity':
                out_of_fold = raw_calibration
            else:
                out_of_fold = np.empty(len(calibration))
                for fit_indices, held_indices in folds:
                    fold_calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=method)
                    fold_calibrator.fit(calibration[columns].iloc[fit_indices], calibration.Class.iloc[fit_indices].to_numpy())
                    out_of_fold[held_indices] = fold_calibrator.predict_proba(calibration[columns].iloc[held_indices])[:, 1]
            method_checks.append({'method': method, 'out_of_fold_brier': float(brier_score_loss(calibration.Class, out_of_fold)),
                                  'out_of_fold_log_loss': float(log_loss(calibration.Class, out_of_fold))})
        chosen_method = min(method_checks, key=lambda row: row['out_of_fold_brier'])['method']
        calibrated = model if chosen_method == 'identity' else CalibratedClassifierCV(FrozenEstimator(model), method=chosen_method)
        if chosen_method != 'identity':
            calibrated.fit(calibration[columns], calibration.Class.to_numpy())
        if not np.allclose(model.predict_proba(selection[columns].iloc[:20])[:, 1], raw_selection[:20]):
            raise ValueError('Calibration changed the base classifier')
        calibrated_selection = calibrated.predict_proba(selection[columns])[:, 1]
        calibrated_calibration = calibrated.predict_proba(calibration[columns])[:, 1]
        canceled = cancel_from_probability(calibrated_selection, selection.Amount, costs)
        metrics = selection_metrics(selection.Class, selection.Amount, calibrated_selection, canceled, costs, True)
        raw_metrics = next(row for row in experiment['results'] if row['name'] == name)
        if not np.isclose(raw_metrics['brier_score'], brier_score_loss(selection.Class, raw_selection)):
            raise ValueError('Raw model metric changed')
        in_sample = {
            'raw_brier': float(brier_score_loss(calibration.Class, raw_calibration)),
            'calibrated_brier': float(brier_score_loss(calibration.Class, calibrated_calibration)),
            'raw_log_loss': float(log_loss(calibration.Class, raw_calibration)),
            'calibrated_log_loss': float(log_loss(calibration.Class, calibrated_calibration)),
            'observed_fraud_rate': float(calibration.Class.mean()),
            'mean_calibrated_probability': float(calibrated_calibration.mean()),
        }
        bundle = {'pipeline': calibrated, 'base_model': name, 'input_columns': columns, 'calibration_method': chosen_method,
                  'policy': 'cancel_if_probability_above_0.20_for_positive_amount',
                  'cost_scenario': costs, 'protocol': protocol,
                  'lineage': {**metadata, 'raw_model_sha256': sha256(raw_bundle_path)},
                  'posthoc_calibrated': chosen_method != 'identity'}
        model_path = output_dir / f'{name}_{chosen_method}.joblib'
        joblib.dump(bundle, model_path)
        restored = joblib.load(model_path)['pipeline']
        if not np.allclose(restored.predict_proba(selection[columns].iloc[:20])[:, 1], calibrated_selection[:20]):
            raise ValueError(f'Calibrated artifact fails reload: {name}')
        with mlflow.start_run(run_name=f'{name}_{chosen_method}') as run:
            mlflow.log_params({'base_model': name, 'calibration_method': chosen_method, 'calibration_split': 'calibration',
                               'comparison_split': 'selection', 'threshold': 0.2, 'calibration_rows': len(calibration),
                               'calibration_frauds': int(calibration.Class.sum())})
            mlflow.log_metrics({f'selection_{key}': value for key, value in metrics.items()
                                if isinstance(value, (int, float)) and value is not None and np.isfinite(value)})
            mlflow.log_metrics({f'calibration_descriptive_{key}': value for key, value in in_sample.items()})
            mlflow.log_metrics({f'calibration_cv_{row["method"]}_brier': row['out_of_fold_brier'] for row in method_checks})
            mlflow.set_tags({key: metadata[key] for key in ['code_sha256', 'assignment_sha256', 'protocol_sha256', 'cost_config_sha256']})
            mlflow.log_dict(metadata, 'lineage.json')
            mlflow.log_artifact(str(model_path), artifact_path='policy_pipeline')
            results.append({'name': name, 'chosen_method': chosen_method, 'calibration_cv': method_checks,
                            'mlflow_run_id': run.info.run_id, 'raw_model_sha256': sha256(raw_bundle_path),
                            'policy_model_sha256': sha256(model_path), 'selection_raw': raw_metrics,
                            'selection_policy': metrics, 'calibration_descriptive': in_sample})
        selection_frames.append(pd.DataFrame({'source_row_id': selection.index, 'model': name,
                                              'raw_probability': raw_selection, 'policy_probability': calibrated_selection,
                                              'canceled': canceled, 'Class': selection.Class.to_numpy(), 'Amount': selection.Amount.to_numpy(),
                                              'simulated_cost': action_costs(selection.Class, selection.Amount, canceled, costs),
                                              'simulated_profit': action_profits(selection.Class, selection.Amount, canceled, costs)}))
        calibration_frames.append(pd.DataFrame({'source_row_id': calibration.index, 'model': name,
                                                'raw_probability': raw_calibration, 'policy_probability': calibrated_calibration,
                                                'Class': calibration.Class.to_numpy()}))
    selection_path = ROOT / 'data/processed/calibrated-selection-predictions.csv'
    calibration_path = ROOT / 'data/processed/calibration-descriptive-predictions.csv'
    pd.concat(selection_frames, ignore_index=True).to_csv(selection_path, index=False)
    pd.concat(calibration_frames, ignore_index=True).to_csv(calibration_path, index=False)
    method_selected_leader = max(results, key=lambda row: row['selection_policy']['simulated_profit_per_1000'])['name']
    all_policies = [(row['name'], variant, row[f'selection_{variant}']['simulated_profit_per_1000'])
                    for row in results for variant in ['raw', 'policy']]
    overall_leader = max(all_policies, key=lambda row: row[2])
    report = {'metadata': metadata, 'results': results,
              'method_selected_policy_profit_leader': method_selected_leader,
              'all_development_policy_profit_leader': {'model': overall_leader[0], 'variant': overall_leader[1]},
              'candidate_frozen': False,
              'selection_predictions': {'path': str(selection_path.relative_to(ROOT)), 'sha256': sha256(selection_path)},
              'calibration_descriptive_predictions': {'path': str(calibration_path.relative_to(ROOT)), 'sha256': sha256(calibration_path)}}
    (ROOT / 'reports/calibration-results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    table = pd.DataFrame([{'model': row['name'], 'raw_ap': row['selection_raw']['average_precision'],
                           'chosen_method': row['chosen_method'],
                           'policy_ap': row['selection_policy']['average_precision'],
                           'raw_brier': row['selection_raw']['brier_score'],
                           'policy_brier': row['selection_policy']['brier_score'],
                           'raw_profit_per_1000': row['selection_raw']['simulated_profit_per_1000'],
                           'policy_profit_per_1000': row['selection_policy']['simulated_profit_per_1000'],
                           'policy_fraud_recall': row['selection_policy']['fraud_canceled_recall'],
                           'policy_legit_cancel_rate': row['selection_policy']['legitimate_cancellation_rate']}
                          for row in results])
    print(table.round(6).to_string(index=False))
    print(f'Method-selected policy leader: {method_selected_leader}; all development variants leader: {overall_leader[:2]}. No candidate frozen. Evaluation was not scored.')


if __name__ == '__main__':
    main()
