"""Audit selection-set errors and model feature usage without refitting."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from run_baselines import ROOT, sha256


def checked_predictions(report, expected_models, assignments):
    path = ROOT / report['selection_predictions']['path']
    if sha256(path) != report['selection_predictions']['sha256']:
        raise ValueError(f'Prediction file changed: {path}')
    predictions = pd.read_csv(path)
    selection_ids = set(assignments.index[assignments['split'].eq('selection')])
    if set(predictions.model) != expected_models:
        raise ValueError('Unexpected model names')
    for name, group in predictions.groupby('model'):
        if len(group) != len(selection_ids) or set(group.source_row_id) != selection_ids or not group.source_row_id.is_unique:
            raise ValueError(f'Selection population mismatch: {name}')
    return predictions


def main():
    baseline_path = ROOT / 'reports/baseline-results.json'
    experiment_path = ROOT / 'reports/experiment-results.json'
    baseline = json.loads(baseline_path.read_text())
    experiment = json.loads(experiment_path.read_text())
    audit = json.loads((ROOT / 'reports/duplicate-split-audit.json').read_text())
    assignments_path = ROOT / audit['assignments']['path']
    if sha256(assignments_path) != audit['assignments']['sha256']:
        raise ValueError('Split assignments changed')
    if experiment['metadata']['baseline_report_sha256'] != sha256(baseline_path):
        raise ValueError('Experiment baseline lineage changed')
    if experiment['metadata']['assignments_sha256'] != audit['assignments']['sha256']:
        raise ValueError('Experiment split lineage changed')
    assignments = pd.read_csv(assignments_path).set_index('source_row_id')
    if not audit['group_disjoint'] or assignments.groupby('pattern_group_id').split.nunique().max() != 1:
        raise ValueError('Pattern groups span splits')
    base = checked_predictions(baseline, {x['name'] for x in baseline['results']}, assignments)
    trees = checked_predictions(experiment, {x['name'] for x in experiment['results']}, assignments)
    predictions = pd.concat([base, trees], ignore_index=True)
    if not predictions.model.isin(['logistic_regression', 'random_forest', 'lightgbm', 'approve_all', 'amount_rule']).all():
        raise ValueError('Unexpected comparator')
    predictions['error_type'] = np.select([
        predictions.Class.eq(1) & ~predictions.canceled,
        predictions.Class.eq(0) & predictions.canceled,
        predictions.Class.eq(1) & predictions.canceled,
    ], ['approved_fraud', 'canceled_legitimate', 'canceled_fraud'], default='approved_legitimate')
    metrics = []
    for name, frame in predictions.groupby('model', sort=False):
        metrics.append({
            'model': name, 'rows': len(frame), 'frauds': int(frame.Class.sum()),
            'approved_frauds': int(frame.error_type.eq('approved_fraud').sum()),
            'approved_fraud_amount': float(frame.loc[frame.error_type.eq('approved_fraud'), 'Amount'].sum()),
            'canceled_legitimate': int(frame.error_type.eq('canceled_legitimate').sum()),
            'forgone_legitimate_margin': float(frame.loc[frame.error_type.eq('canceled_legitimate'), 'simulated_cost'].sum()),
            'canceled_frauds': int(frame.error_type.eq('canceled_fraud').sum()),
            'canceled_fraud_amount': float(frame.loc[frame.error_type.eq('canceled_fraud'), 'Amount'].sum()),
            'total_opportunity_loss': float(frame.simulated_cost.sum()),
            'top_5_error_loss_share_pct': float(100 * frame.simulated_cost.nlargest(5).sum() / frame.simulated_cost.sum()),
        })
    summary = pd.DataFrame(metrics)
    details = predictions.loc[predictions.error_type.isin(['approved_fraud', 'canceled_legitimate'])].copy()
    details['Time'] = assignments.loc[details.source_row_id, 'Time'].to_numpy()
    details = details.sort_values(['simulated_cost', 'source_row_id'], ascending=[False, True])
    details_path = ROOT / 'data/processed/selection-expensive-errors.csv'
    details.to_csv(details_path, index=False)
    learned = predictions.loc[predictions.model.isin(['logistic_regression', 'random_forest', 'lightgbm'])]
    wide = learned.pivot(index='source_row_id', columns='model', values=['score', 'canceled', 'Class', 'Amount'])
    for field in ['Class', 'Amount']:
        if not ((wide[(field, 'logistic_regression')] == wide[(field, 'random_forest')]) &
                (wide[(field, 'random_forest')] == wide[(field, 'lightgbm')])).all():
            raise ValueError(f'Labels or amounts differ across models: {field}')
    cases = pd.DataFrame(index=wide.index)
    cases['Class'] = wide[('Class', 'random_forest')].astype(int)
    cases['Amount'] = wide[('Amount', 'random_forest')].astype(float)
    cases['Time'] = assignments.loc[cases.index, 'Time'].to_numpy()
    cases['pattern_group_id'] = assignments.loc[cases.index, 'pattern_group_id'].to_numpy()
    for name in ['logistic_regression', 'random_forest', 'lightgbm']:
        cases[f'{name}_score'] = wide[('score', name)].astype(float)
        cases[f'{name}_canceled'] = wide[('canceled', name)].astype(bool)
    rf, lgb = cases.random_forest_canceled, cases.lightgbm_canceled
    categories = {
        'fraud_missed_by_both_trees': cases.Class.eq(1) & ~rf & ~lgb,
        'fraud_caught_by_both_trees': cases.Class.eq(1) & rf & lgb,
        'fraud_tree_decisions_disagree': cases.Class.eq(1) & rf.ne(lgb),
        'legitimate_canceled_by_either_tree': cases.Class.eq(0) & (rf | lgb),
    }
    case_rows = []
    category_counts = {}
    for category, mask in categories.items():
        subset = cases.loc[mask].sort_values('Amount', ascending=False)
        category_counts[category] = len(subset)
        selected = subset.drop_duplicates('pattern_group_id').head(5).copy()
        selected.insert(0, 'category', category)
        case_rows.append(selected)
    case_path = ROOT / 'data/processed/selection-case-studies.csv'
    case_table = pd.concat(case_rows).rename_axis('source_row_id').reset_index()
    case_table.to_csv(case_path, index=False)
    importance_rows = []
    model_artifacts = {}
    columns = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
    for name in ['random_forest', 'lightgbm']:
        model_path = ROOT / f'models/experiments/{name}.joblib'
        model_artifacts[name] = sha256(model_path)
        bundle = joblib.load(model_path)
        if bundle['input_columns'] != columns or bundle['lineage']['code_sha256'] != experiment['metadata']['code_sha256']:
            raise ValueError(f'Stale model bundle: {name}')
        model = bundle['model']
        values = model.feature_importances_ if name == 'random_forest' else model.booster_.feature_importance(importance_type='gain')
        for feature, value in zip(columns, values, strict=True):
            importance_rows.append({'model': name, 'feature': feature, 'importance_type': 'impurity' if name == 'random_forest' else 'gain',
                                    'relative_importance_pct': float(100 * value / values.sum())})
    importance = pd.DataFrame(importance_rows).sort_values(['model', 'relative_importance_pct'], ascending=[True, False])
    importance_path = ROOT / 'reports/feature-importance-selection.csv'
    importance.to_csv(importance_path, index=False)
    # Explain selected LightGBM scores in raw log-odds, using predictors only.
    wanted = set(case_table.source_row_id)
    case_parts, offset = [], 0
    for source in audit['source_files']:
        source_path = ROOT / source['path']
        if sha256(source_path) != source['sha256']:
            raise ValueError(f'Source changed: {source_path}')
        for chunk in pd.read_csv(source_path, usecols=columns, chunksize=50000):
            chunk.index = np.arange(offset, offset + len(chunk))
            offset += len(chunk)
            case_parts.append(chunk.loc[chunk.index.isin(wanted)])
    case_features = pd.concat(case_parts).loc[case_table.source_row_id, columns]
    lgb_model = joblib.load(ROOT / 'models/experiments/lightgbm.joblib')['model']
    contributions = np.asarray(lgb_model.predict(case_features, pred_contrib=True))
    raw_scores = np.asarray(lgb_model.predict(case_features, raw_score=True))
    probabilities = lgb_model.predict_proba(case_features)[:, 1]
    if contributions.shape != (len(case_table), len(columns) + 1):
        raise ValueError('Unexpected LightGBM contribution shape')
    if not np.allclose(contributions.sum(axis=1), raw_scores, atol=1e-8):
        raise ValueError('Contributions do not reconstruct raw scores')
    if not np.allclose(probabilities, case_table.lightgbm_score, atol=1e-10):
        raise ValueError('Case predictions differ from saved selection predictions')
    local_rows = []
    for row_number, case in case_table.iterrows():
        top_features = np.argsort(np.abs(contributions[row_number, :-1]))[-5:][::-1]
        for rank, feature_number in enumerate(top_features, start=1):
            feature = columns[feature_number]
            local_rows.append({'source_row_id': int(case.source_row_id), 'category': case.category,
                               'rank': rank, 'feature': feature, 'feature_value': float(case_features.iloc[row_number][feature]),
                               'log_odds_contribution': float(contributions[row_number, feature_number]),
                               'base_log_odds': float(contributions[row_number, -1]),
                               'full_raw_log_odds': float(raw_scores[row_number]),
                               'predicted_probability': float(probabilities[row_number])})
    local_path = ROOT / 'reports/selection-lightgbm-local-contributions.csv'
    pd.DataFrame(local_rows).to_csv(local_path, index=False)
    report = {
        'scope': 'selection only; no calibration or evaluation labels used',
        'code_sha256': sha256(Path(__file__)),
        'baseline_report_sha256': sha256(baseline_path),
        'experiment_report_sha256': sha256(experiment_path),
        'assignment_sha256': audit['assignments']['sha256'],
        'model_artifact_sha256': model_artifacts,
        'models': metrics,
        'case_category_counts': category_counts,
        'expensive_errors': {'path': str(details_path.relative_to(ROOT)), 'sha256': sha256(details_path)},
        'case_studies': {'path': str(case_path.relative_to(ROOT)), 'sha256': sha256(case_path)},
        'feature_importance': {'path': str(importance_path.relative_to(ROOT)), 'sha256': sha256(importance_path)},
        'lightgbm_local_contributions': {'path': str(local_path.relative_to(ROOT)), 'sha256': sha256(local_path),
                                         'units': 'raw log-odds; five largest absolute contributions per case plus base value'},
        'limitations': ['Native feature importance describes fitted-tree usage and is not causal.',
                        'LightGBM local contributions decompose model raw scores; they do not establish why a payment happened or why it was fraudulent.',
                        'Top errors and importance are descriptive selection-set findings; probabilities are uncalibrated.',
                        'PCA component meanings are unavailable; no merchant or customer narrative is inferred.'],
    }
    (ROOT / 'reports/failure-analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(summary.round(3).to_string(index=False))
    print('\nCase categories:', category_counts)
    print('\nTop 5 LightGBM gain features:')
    print(importance.loc[importance.model.eq('lightgbm')].head(5).round(2).to_string(index=False))


if __name__ == '__main__':
    main()
