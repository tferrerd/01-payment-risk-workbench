"""Freeze a selection-chosen policy, then cache its single final evaluation."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from run_baselines import ROOT, action_costs, action_profits, cancel_from_probability, setup_tracking, sha256


BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 42


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def paired_group_bootstrap(profits, groups, candidate, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED):
    """Resample whole patterns uniformly; divide each draw by its sampled row count."""
    values = np.asarray(profits, dtype=float)
    groups = np.asarray(groups)
    if values.ndim != 2 or len(values) != len(groups) or not len(values) or not np.isfinite(values).all() or pd.isna(groups).any():
        raise ValueError('Expected finite policy profits and one nonmissing group per row')
    if repeats < 2 or candidate not in profits.columns:
        raise ValueError('Expected at least two replicates and a candidate column')
    codes, unique = pd.factorize(groups)
    group_sizes = np.bincount(codes)
    totals = np.zeros((len(unique), values.shape[1]))
    np.add.at(totals, codes, values)
    rng = np.random.default_rng(seed)
    draws = np.empty((repeats, values.shape[1]))
    for start in range(0, repeats, 16):
        sampled = rng.integers(0, len(unique), size=(min(16, repeats-start), len(unique)))
        denominators = group_sizes[sampled].sum(axis=1)
        for column in range(values.shape[1]):
            draws[start:start+len(sampled), column] = 1000 * totals[sampled, column].sum(axis=1) / denominators
    point = 1000 * values.mean(axis=0)
    intervals, differences = [], []
    candidate_column = profits.columns.get_loc(candidate)
    for column, name in enumerate(profits.columns):
        low, high = np.quantile(draws[:, column], [0.025, 0.975])
        intervals.append({'policy': name, 'profit_per_1000': float(point[column]), 'lower_95': float(low), 'upper_95': float(high)})
        if name != candidate:
            low, high = np.quantile(draws[:, candidate_column] - draws[:, column], [0.025, 0.975])
            differences.append({'candidate': candidate, 'comparator': name,
                                'profit_difference_per_1000': float(point[candidate_column] - point[column]),
                                'lower_95': float(low), 'upper_95': float(high),
                                'conclusion': 'positive' if low > 0 else 'negative' if high < 0 else 'inconclusive'})
    return {'rows': len(values), 'groups': len(unique), 'repeats': repeats, 'seed': seed,
            'method': 'Unstratified paired pattern-group bootstrap, percentile 95% intervals; ratio uses sampled payment count',
            'policy_intervals': intervals, 'paired_differences': differences}, pd.DataFrame(draws, columns=profits.columns)


def policy_metrics(labels, amounts, scores, canceled, costs, probability=True):
    labels, amounts, scores, canceled = map(np.asarray, [labels, amounts, scores, canceled])
    profit = action_profits(labels, amounts, canceled, costs)
    oracle = float((costs['legitimate_margin_fraction'] * amounts * (1-labels)).sum())
    return {'rows': len(labels), 'frauds': int(labels.sum()),
            'average_precision': float(average_precision_score(labels, scores)),
            'roc_auc': float(roc_auc_score(labels, scores)),
            'brier_score': float(brier_score_loss(labels, scores)) if probability else None,
            'frauds_canceled': int(canceled[labels == 1].sum()),
            'fraud_canceled_recall': float(canceled[labels == 1].mean()),
            'legitimate_canceled': int(canceled[labels == 0].sum()),
            'legitimate_cancellation_rate': float(canceled[labels == 0].mean()),
            'cancellation_rate': float(canceled.mean()), 'simulated_total_profit': float(profit.sum()),
            'simulated_profit_per_1000': float(profit.mean()*1000), 'maximum_possible_profit': oracle,
            'profit_pct_of_maximum': float(100*profit.sum()/oracle),
            'total_opportunity_loss': float(action_costs(labels, amounts, canceled, costs).sum())}


def checked_report(path, assignments, split='selection'):
    report = json.loads(path.read_text())
    item = report['selection_predictions']
    prediction_path = ROOT / item['path']
    if sha256(prediction_path) != item['sha256']:
        raise ValueError(f'Predictions changed: {prediction_path}')
    data = pd.read_csv(prediction_path)
    expected_ids = set(assignments.index[assignments.split.eq(split)])
    for name, frame in data.groupby('model'):
        if not frame.source_row_id.is_unique or set(frame.source_row_id) != expected_ids:
            raise ValueError(f'Unexpected selection rows for {name}')
    return report, data


def prepare(assignments, audit, costs, protocol, lineage):
    freeze_path = ROOT / 'config/candidate-freeze.json'
    selection_path = ROOT / 'reports/selection-uncertainty.json'
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text())
        for path, expected in freeze['lineage'].items():
            if sha256(ROOT / path) != expected:
                raise ValueError(f'Frozen input changed: {path}; do not overwrite the final evaluation')
        if sha256(ROOT / selection_path) != freeze['selection_uncertainty_sha256']:
            raise ValueError('Frozen selection uncertainty changed')
        for item in [freeze['candidate'], freeze['logistic_reference']]:
            if sha256(ROOT / item['snapshot_path']) != item['snapshot_sha256']:
                raise ValueError('Frozen model snapshot changed')
        return freeze
    if selection_path.exists():
        raise ValueError('Selection report exists without freeze; inspect partial work before continuing')
    baseline, base = checked_report(ROOT / 'reports/baseline-results.json', assignments)
    experiment, trees = checked_report(ROOT / 'reports/experiment-results.json', assignments)
    calibration, mapped = checked_report(ROOT / 'reports/calibration-results.json', assignments)
    if experiment['metadata']['baseline_report_sha256'] != sha256(ROOT / 'reports/baseline-results.json') or calibration['metadata']['experiment_report_sha256'] != sha256(ROOT / 'reports/experiment-results.json'):
        raise ValueError('Development report lineage changed')
    for report in [baseline, experiment, calibration]:
        if report['metadata']['cost_config_sha256'] != sha256(ROOT / 'config/cost-scenario.json') or report['metadata']['protocol_sha256'] != sha256(ROOT / 'config/evaluation.json'):
            raise ValueError('Development economics or protocol changed')
    frames = {name: group.set_index('source_row_id').sort_index() for name, group in base.groupby('model')}
    metrics = {row['name']: row for row in baseline['results']}
    artifacts = {}
    for row in calibration['results']:
        name = row['name']
        frame = trees.loc[trees.model.eq(name)].set_index('source_row_id').sort_index()
        frames[f'{name}_raw'] = frame
        metrics[f'{name}_raw'] = row['selection_raw']
        raw_path = ROOT / f'models/experiments/{name}.joblib'
        if sha256(raw_path) != row['raw_model_sha256']:
            raise ValueError(f'Base artifact changed: {name}')
        artifacts[f'{name}_raw'] = (raw_path, 'model', 'identity')
        if row['chosen_method'] != 'identity':
            variant = f"{name}_{row['chosen_method']}"
            frames[variant] = mapped.loc[mapped.model.eq(name)].set_index('source_row_id').sort_index()
            metrics[variant] = row['selection_policy']
            path = ROOT / f"models/calibrated/{variant}.joblib"
            if sha256(path) != row['policy_model_sha256']:
                raise ValueError(f'Calibrated artifact changed: {name}')
            artifacts[variant] = (path, 'pipeline', row['chosen_method'])
    artifacts['logistic_regression'] = (ROOT / 'models/baselines/logistic_regression.joblib', 'pipeline', 'identity')
    reference = frames['approve_all']
    profits = pd.DataFrame(index=reference.index)
    for name, frame in frames.items():
        if not frame.index.equals(reference.index) or not np.array_equal(frame.Class, reference.Class) or not np.array_equal(frame.Amount, reference.Amount):
            raise ValueError('Selection populations differ across policies')
        profit = action_profits(frame.Class, frame.Amount, frame.canceled, costs)
        if not np.allclose(profit, frame.simulated_profit):
            raise ValueError('Stored selection economics differ')
        profits[name] = profit
    candidate = max(sorted(artifacts), key=lambda name: profits[name].mean())
    uncertainty, draws = paired_group_bootstrap(profits, assignments.loc[profits.index, 'pattern_group_id'], candidate)
    uncertainty.update({'split': 'selection', 'candidate_by_point_estimate': candidate,
                        'selection_rule': 'Maximum observed primary-scenario selection profit among fitted learned alternatives; intervals describe uncertainty and do not prove superiority',
                        'frauds': int(reference.Class.sum()),
                        'limitations': 'Conditional on fixed models and observed exact-pattern groups; does not include training, method-selection, or repeated-development-search uncertainty.'})
    draw_path = ROOT / 'data/processed/selection-bootstrap-profit.csv'
    draws.to_csv(draw_path, index=False)
    write_json(selection_path, uncertainty)
    original_path, model_key, mapping = artifacts[candidate]
    original = joblib.load(original_path)
    logistic_original = joblib.load(ROOT / 'models/baselines/logistic_regression.joblib')
    if logistic_original['lineage']['code_sha256'] != baseline['metadata']['code_sha256']:
        raise ValueError('Logistic reference lineage changed')
    columns = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
    if original['input_columns'] != columns:
        raise ValueError('Candidate feature schema changed')
    threshold = costs['legitimate_margin_fraction'] / (costs['legitimate_margin_fraction'] + costs['fraud_loss_fraction'])
    snapshot = {'pipeline': original[model_key], 'input_columns': columns, 'calibration_method': mapping,
                'posthoc_calibrated': mapping != 'identity', 'cost_scenario': costs,
                'policy': {'threshold': threshold, 'tie': 'approve', 'positive_amount_only': True}, 'lineage': lineage}
    frozen_dir = ROOT / 'models/frozen'
    frozen_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = frozen_dir / 'candidate-v1.joblib'
    logistic_path = frozen_dir / 'logistic-reference-v1.joblib'
    joblib.dump(snapshot, candidate_path)
    shutil.copy2(ROOT / 'models/baselines/logistic_regression.joblib', logistic_path)
    freeze = {'version': 'candidate-v1', 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'evaluation_scored_when_frozen': False,
              'candidate': {'name': candidate, 'mapping': mapping, 'posthoc_calibrated': mapping != 'identity',
                            'snapshot_path': str(candidate_path.relative_to(ROOT)), 'snapshot_sha256': sha256(candidate_path)},
              'logistic_reference': {'snapshot_path': str(logistic_path.relative_to(ROOT)), 'snapshot_sha256': sha256(logistic_path)},
              'lineage': lineage, 'selection_uncertainty_sha256': sha256(selection_path),
              'decision_rule': uncertainty['selection_rule'], 'primary_comparator': 'approve_all',
              'secondary_comparators': ['amount_rule', 'logistic_regression'],
              'threshold': threshold, 'training_prior': baseline['metadata']['training_prior'],
              'training_amount_p99': baseline['metadata']['amount_rule_threshold_from_train_p99'],
              'frozen_fpr_diagnostic': {'threshold': metrics[candidate]['selection_fpr_diagnostic_threshold'],
                                       'predict_none': metrics[candidate]['selection_fpr_diagnostic_predict_none'],
                                       'selection_target_fpr': 0.001, 'comparison': 'score >= threshold'},
              'sensitivity_plan': {'fractions': costs['sensitivity'],
                                   'policies': ['hold primary actions fixed', 'use margin/(margin+fraud_loss) threshold on frozen scores'],
                                   'use': 'Descriptive scenarios only; no candidate or threshold chosen using evaluation outcomes'},
              'bootstrap': {'repeats': BOOTSTRAP_REPEATS, 'seed': BOOTSTRAP_SEED, 'unit': 'exact V1–V28/Amount pattern group'},
              'evidence_rule': 'Primary improvement is supported only if the paired 95% interval vs approve_all is wholly positive; otherwise inconclusive or negative. No production promotion is implied.',
              'evaluation_exposure': protocol['evaluation_exposure']}
    write_json(freeze_path, freeze)
    print('Frozen before evaluation:', candidate, flush=True)
    print(pd.DataFrame(uncertainty['paired_differences']).round(3).to_string(index=False), flush=True)
    return freeze


def evaluate(freeze, assignments, audit, costs, protocol):
    report_path = ROOT / 'reports/final-evaluation.json'
    freeze_path = ROOT / 'config/candidate-freeze.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['metadata']['freeze_sha256'] != sha256(freeze_path):
            raise ValueError('Existing final evaluation belongs to a different freeze')
        path = ROOT / report['predictions']['path']
        if sha256(path) != report['predictions']['sha256']:
            raise ValueError('Cached evaluation predictions changed')
        print('Using cached final evaluation; no new scoring or model selection performed.')
        return report
    parts, offset = [], 0
    for source in audit['source_files']:
        for chunk in pd.read_csv(ROOT / source['path'], chunksize=50000):
            chunk.index = np.arange(offset, offset+len(chunk))
            offset += len(chunk)
            parts.append(chunk.loc[assignments.loc[chunk.index, 'split'].eq('evaluation').to_numpy()])
    evaluation = pd.concat(parts)
    expected = next(row for row in audit['split_summary'] if row['split'] == 'evaluation')
    if len(evaluation) != expected['rows'] or int(evaluation.Class.sum()) != expected['frauds'] or not np.isfinite(evaluation.to_numpy()).all():
        raise ValueError('Unexpected evaluation population')
    candidate_bundle = joblib.load(ROOT / freeze['candidate']['snapshot_path'])
    logistic_bundle = joblib.load(ROOT / freeze['logistic_reference']['snapshot_path'])
    columns = candidate_bundle['input_columns']
    candidate = freeze['candidate']['name']
    score = candidate_bundle['pipeline'].predict_proba(evaluation[columns])[:, 1]
    y, amount = evaluation.Class.to_numpy(), evaluation.Amount.to_numpy()
    policy_scores = {candidate: score, 'approve_all': np.full(len(evaluation), freeze['training_prior']),
                     'amount_rule': amount.copy(),
                     'logistic_regression': logistic_bundle['pipeline'].predict_proba(evaluation[columns])[:, 1]}
    actions = {name: np.zeros(len(y), dtype=bool) if name == 'approve_all' else amount >= freeze['training_amount_p99'] if name == 'amount_rule'
               else cancel_from_probability(values, amount, costs) for name, values in policy_scores.items()}
    results = [{'name': name, **policy_metrics(y, amount, values, actions[name], costs, name != 'amount_rule')}
               for name, values in policy_scores.items()]
    profits = pd.DataFrame({name: action_profits(y, amount, actions[name], costs) for name in policy_scores}, index=evaluation.index)
    uncertainty, draws = paired_group_bootstrap(profits, assignments.loc[evaluation.index, 'pattern_group_id'], candidate)
    draws.to_csv(ROOT / 'data/processed/evaluation-bootstrap-profit.csv', index=False)
    primary = next(row for row in uncertainty['paired_differences'] if row['comparator'] == 'approve_all')
    sensitivity = []
    for margin in costs['sensitivity']['legitimate_margin_fraction']:
        for loss in costs['sensitivity']['fraud_loss_fraction']:
            scenario = {**costs, 'legitimate_margin_fraction': margin, 'fraud_loss_fraction': loss}
            approve_all = action_profits(y, amount, np.zeros(len(y), dtype=bool), scenario)
            scenario_action = cancel_from_probability(score, amount, scenario)
            fixed = action_profits(y, amount, actions[candidate], scenario)
            scenario_profit = action_profits(y, amount, scenario_action, scenario)
            sensitivity.append({'margin_fraction': margin, 'fraud_loss_fraction': loss, 'scenario_threshold': margin/(margin+loss),
                                'primary_scenario': margin == costs['legitimate_margin_fraction'] and loss == costs['fraud_loss_fraction'],
                                'frozen_actions_profit_difference_per_1000': float(1000*(fixed-approve_all).mean()),
                                'scenario_threshold_profit_difference_per_1000': float(1000*(scenario_profit-approve_all).mean()),
                                'scenario_cancellation_rate': float(scenario_action.mean())})
    diagnostic = freeze['frozen_fpr_diagnostic']
    flagged = np.zeros(len(y), dtype=bool) if diagnostic['predict_none'] else score >= diagnostic['threshold']
    loss = action_costs(y, amount, actions[candidate], costs)
    prediction_path = ROOT / 'data/processed/final-evaluation-predictions.csv'
    pd.concat([pd.DataFrame({'source_row_id': evaluation.index, 'model': name, 'score': values, 'canceled': actions[name],
                            'Class': y, 'Amount': amount, 'simulated_profit': profits[name].to_numpy()})
               for name, values in policy_scores.items()], ignore_index=True).to_csv(prediction_path, index=False)
    report = {'metadata': {'evaluated_split': 'evaluation', 'rows': len(y), 'frauds': int(y.sum()),
                           'scored_at_utc': datetime.now(timezone.utc).isoformat(), 'freeze_sha256': sha256(freeze_path),
                           'code_sha256': sha256(Path(__file__)), 'candidate': candidate, 'lineage': freeze['lineage'],
                           'exposure_limitation': protocol['evaluation_exposure'],
                           'interval_scope': 'Conditional on fitted artifacts and observed pattern groups; not training uncertainty, long-term drift, or real business impact.'},
              'results': results, 'uncertainty': uncertainty, 'sensitivity': sensitivity,
              'frozen_fpr_diagnostic': {**diagnostic, 'evaluation_fraud_recall': float(flagged[y == 1].mean()),
                                       'evaluation_realized_fpr': float(flagged[y == 0].mean())},
              'opportunity_loss_concentration': {'largest_error': float(loss.max()), 'top_5_loss_share_pct': float(100*np.sort(loss)[-5:].sum()/loss.sum()) if loss.sum() else 0.0},
              'primary_evidence_conclusion': primary['conclusion'],
              'recommendation': 'Offline modeled gain is supported under this scenario; proceed to a versioned prototype, with calibration and source-population limitations disclosed.' if primary['conclusion'] == 'positive'
                                else 'Do not claim a demonstrated profit improvement or promote for production; primary paired evidence is inconclusive or negative. Retain an offline prototype and seek more labeled evidence.',
              'predictions': {'path': str(prediction_path.relative_to(ROOT)), 'sha256': sha256(prediction_path)}}
    write_json(report_path, report)
    setup_tracking('07-frozen-evaluation')
    with mlflow.start_run(run_name=f'{candidate}_candidate_v1') as run:
        mlflow.log_params({'candidate': candidate, 'split': 'evaluation', 'threshold': freeze['threshold'],
                           'bootstrap_repeats': BOOTSTRAP_REPEATS, 'bootstrap_seed': BOOTSTRAP_SEED, 'posthoc_calibrated': freeze['candidate']['posthoc_calibrated']})
        candidate_metrics = results[0]
        mlflow.log_metrics({key: value for key, value in candidate_metrics.items() if isinstance(value, (int, float)) and value is not None})
        mlflow.log_metrics({'profit_difference_vs_approve_all_per_1000': primary['profit_difference_per_1000'],
                            'profit_difference_lower_95': primary['lower_95'], 'profit_difference_upper_95': primary['upper_95']})
        mlflow.set_tag('freeze_sha256', sha256(freeze_path))
        mlflow.log_artifact(str(freeze_path), artifact_path='freeze')
        mlflow.log_dict(report, 'final-evaluation.json')
        report['mlflow_run_id'] = run.info.run_id
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true', help='Compute selection uncertainty and freeze without reading evaluation labels')
    args = parser.parse_args()
    audit = json.loads((ROOT / 'reports/duplicate-split-audit.json').read_text())
    costs = json.loads((ROOT / 'config/cost-scenario.json').read_text())
    protocol = json.loads((ROOT / 'config/evaluation.json').read_text())
    paths = ['config/evaluation.json', 'config/cost-scenario.json', 'reports/duplicate-split-audit.json',
             'reports/baseline-results.json', 'reports/experiment-results.json', 'reports/calibration-results.json',
             'data/processed/split-assignments.csv', 'poetry.lock', 'scripts/run_final_evaluation.py', 'scripts/run_baselines.py']
    lineage = {path: sha256(ROOT / path) for path in paths}
    if lineage['config/evaluation.json'] != audit['config_sha256'] or lineage['data/processed/split-assignments.csv'] != audit['assignments']['sha256']:
        raise ValueError('Audit or assignments changed')
    for source in audit['source_files']:
        if sha256(ROOT / source['path']) != source['sha256']:
            raise ValueError('Source data changed')
    assignments = pd.read_csv(ROOT / 'data/processed/split-assignments.csv').set_index('source_row_id')
    if not audit['group_disjoint'] or assignments.groupby('pattern_group_id').split.nunique().max() != 1:
        raise ValueError('Pattern leakage across splits')
    freeze = prepare(assignments, audit, costs, protocol, lineage)
    if args.prepare_only:
        print('Preparation complete; evaluation has not been scored by this command.')
        return
    report = evaluate(freeze, assignments, audit, costs, protocol)
    print(pd.DataFrame(report['results'])[['name', 'average_precision', 'brier_score', 'simulated_profit_per_1000', 'fraud_canceled_recall', 'legitimate_canceled']].round(6).to_string(index=False))
    print(pd.DataFrame(report['uncertainty']['paired_differences']).round(3).to_string(index=False))
    print(report['recommendation'])


if __name__ == '__main__':
    main()
