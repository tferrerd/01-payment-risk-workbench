"""Reprice existing predictions at a 0.3% legitimate margin; fit no models."""
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from run_baselines import ROOT, action_profits, cancel_from_probability, setup_tracking, sha256
from run_final_evaluation import checked_report, paired_group_bootstrap, write_json


def summarize(labels, amounts, canceled, costs):
    labels, amounts, canceled = map(np.asarray, [labels, amounts, canceled])
    profit = action_profits(labels, amounts, canceled, costs)
    oracle = float((costs['legitimate_margin_fraction'] * amounts * (1-labels)).sum())
    return {'rows': len(labels), 'frauds': int(labels.sum()), 'frauds_canceled': int(canceled[labels == 1].sum()),
            'fraud_canceled_recall': float(canceled[labels == 1].mean()),
            'legitimate_canceled': int(canceled[labels == 0].sum()),
            'legitimate_cancellation_rate': float(canceled[labels == 0].mean()),
            'approved_fraud_amount': float(amounts[(labels == 1) & ~canceled].sum()),
            'forgone_legitimate_margin': float((costs['legitimate_margin_fraction'] * amounts[(labels == 0) & canceled]).sum()),
            'simulated_total_profit': float(profit.sum()), 'simulated_profit_per_1000': float(profit.mean()*1000),
            'maximum_possible_profit': oracle, 'profit_pct_of_maximum': float(100*profit.sum()/oracle)}


def print_results(report):
    for scope, result in report['comparisons'].items():
        print('\n' + scope)
        print(pd.DataFrame(result['results'])[['policy', 'simulated_profit_per_1000', 'frauds_canceled', 'legitimate_canceled']].round(3).to_string(index=False))
        print(pd.DataFrame(result['uncertainty']['paired_differences']).round(3).to_string(index=False))


def main():
    config_path = ROOT / 'config/cost-scenario-low-margin.json'
    costs = json.loads(config_path.read_text())
    threshold = costs['legitimate_margin_fraction'] / (costs['legitimate_margin_fraction'] + costs['fraud_loss_fraction'])
    freeze = json.loads((ROOT / 'config/candidate-freeze.json').read_text())
    final = json.loads((ROOT / 'reports/final-evaluation.json').read_text())
    if final['metadata']['freeze_sha256'] != sha256(ROOT / 'config/candidate-freeze.json'):
        raise ValueError('Historical evaluation freeze changed')
    for path, expected in freeze['lineage'].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f'Historical frozen input changed: {path}')
    audit = json.loads((ROOT / 'reports/duplicate-split-audit.json').read_text())
    assignments = pd.read_csv(ROOT / audit['assignments']['path']).set_index('source_row_id')
    baseline, base = checked_report(ROOT / 'reports/baseline-results.json', assignments)
    experiment, trees = checked_report(ROOT / 'reports/experiment-results.json', assignments)
    calibration, mapped = checked_report(ROOT / 'reports/calibration-results.json', assignments)
    eval_path = ROOT / final['predictions']['path']
    if sha256(eval_path) != final['predictions']['sha256']:
        raise ValueError('Historical evaluation predictions changed')
    evaluation = pd.read_csv(eval_path)
    expected_ids = set(assignments.index[assignments.split.eq('evaluation')])
    for name, frame in evaluation.groupby('model'):
        if not frame.source_row_id.is_unique or set(frame.source_row_id) != expected_ids:
            raise ValueError(f'Unexpected evaluation rows: {name}')
    inputs = ['config/candidate-freeze.json', 'config/cost-scenario.json', 'reports/final-evaluation.json',
              'reports/baseline-results.json', 'reports/experiment-results.json', 'reports/calibration-results.json',
              'data/processed/split-assignments.csv', final['predictions']['path'],
              baseline['selection_predictions']['path'], experiment['selection_predictions']['path'], calibration['selection_predictions']['path']]
    historical_hashes = {path: sha256(ROOT / path) for path in inputs}
    lineage = {**historical_hashes, 'config/cost-scenario-low-margin.json': sha256(config_path),
               'scripts/run_low_margin_scenario.py': sha256(Path(__file__))}
    report_path = ROOT / 'reports/low-margin-scenario.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['metadata']['lineage'] == lineage:
            for comparison in report['comparisons'].values():
                item = comparison['predictions']
                if sha256(ROOT / item['path']) != item['sha256']:
                    raise ValueError('Cached scenario predictions changed')
            print('Using cached scenario analysis of existing scores. No fresh evaluation or model scoring.')
            print_results(report)
            return
    selection_scores = {name: frame.set_index('source_row_id').sort_index() for name, frame in base.groupby('model')}
    selection_scores.update({f'{name}_raw': frame.set_index('source_row_id').sort_index() for name, frame in trees.groupby('model')})
    evaluation_scores = {name: frame.set_index('source_row_id').sort_index() for name, frame in evaluation.groupby('model')}
    metadata = {'scenario': costs['version'], 'legitimate_margin_fraction': costs['legitimate_margin_fraction'],
                'fraud_loss_fraction': costs['fraud_loss_fraction'], 'break_even_threshold': threshold,
                'original_threshold': freeze['threshold'], 'lineage': lineage, 'models_refitted': False,
                'independent_final_evaluation': False, 'existing_candidate': freeze['candidate']['name'],
                'scope': 'New economic hypothesis on existing predictions; selection is development and consumed evaluation is regression only.',
                'notes': costs['limitations']}
    report = {'metadata': metadata, 'comparisons': {}}
    setup_tracking('08-low-margin-economics')
    for scope, scores in [('selection_development', selection_scores), ('evaluation_regression', evaluation_scores)]:
        reference = scores['approve_all']
        ids, labels, amounts = reference.index, reference.Class.to_numpy(), reference.Amount.to_numpy()
        candidate_score = scores['random_forest_raw'].loc[ids, 'score'].to_numpy()
        for name, frame in scores.items():
            if not frame.index.equals(ids) or not np.array_equal(labels, frame.Class) or not np.array_equal(amounts, frame.Amount):
                raise ValueError(f'Population mismatch: {scope}/{name}')
        policies = {
            'approve_all': np.zeros(len(ids), dtype=bool),
            'cancel_all_positive_amounts': amounts > 0,
            'amount_rule': amounts >= freeze['training_amount_p99'],
            'random_forest_original_20pct': (candidate_score > freeze['threshold']) & (amounts > 0),
            'random_forest_break_even': cancel_from_probability(candidate_score, amounts, costs),
            'logistic_break_even': cancel_from_probability(scores['logistic_regression'].loc[ids, 'score'], amounts, costs),
        }
        if scope == 'selection_development':
            policies['lightgbm_break_even'] = cancel_from_probability(scores['lightgbm_raw'].loc[ids, 'score'], amounts, costs)
            iso = mapped.loc[mapped.model.eq('random_forest')].set_index('source_row_id').loc[ids]
            if not np.array_equal(iso.Class, labels) or not np.array_equal(iso.Amount, amounts):
                raise ValueError('Mapped-score selection population changed')
            policies['random_forest_isotonic_break_even'] = cancel_from_probability(iso.policy_probability, amounts, costs)
        profits = pd.DataFrame({name: action_profits(labels, amounts, canceled, costs) for name, canceled in policies.items()}, index=ids)
        results = [{'policy': name, **summarize(labels, amounts, canceled, costs)} for name, canceled in policies.items()]
        uncertainty, draws = paired_group_bootstrap(profits, assignments.loc[ids, 'pattern_group_id'], 'random_forest_break_even')
        prediction_path = ROOT / f'data/processed/low-margin-{scope}-predictions.csv'
        pd.concat([pd.DataFrame({'source_row_id': ids, 'policy': name, 'canceled': canceled,
                                'Class': labels, 'Amount': amounts, 'simulated_profit': profits[name].to_numpy()})
                   for name, canceled in policies.items()], ignore_index=True).to_csv(prediction_path, index=False)
        draws.to_csv(ROOT / f'data/processed/low-margin-{scope}-bootstrap.csv', index=False)
        result = {'results': results, 'uncertainty': uncertainty,
                  'predictions': {'path': str(prediction_path.relative_to(ROOT)), 'sha256': sha256(prediction_path)}}
        with mlflow.start_run(run_name=scope) as run:
            mlflow.log_params({'scenario': costs['version'], 'scope': scope, 'margin_fraction': costs['legitimate_margin_fraction'],
                               'fraud_loss_fraction': costs['fraud_loss_fraction'], 'break_even_threshold': threshold,
                               'models_refitted': False, 'independent_final_evaluation': False})
            for row in results:
                mlflow.log_metrics({row['policy'] + '_' + key: value for key, value in row.items() if isinstance(value, (int, float))})
            mlflow.log_artifact(str(config_path), artifact_path='scenario')
            mlflow.log_dict(metadata, 'lineage.json')
            mlflow.log_dict(result, 'economic-comparison.json')
            result['mlflow_run_id'] = run.info.run_id
        report['comparisons'][scope] = result
    if any(sha256(ROOT / path) != expected for path, expected in historical_hashes.items()):
        raise ValueError('Historical artifact changed during analysis')
    write_json(report_path, report)
    print(f'New break-even probability: {threshold:.10f} ({100*threshold:.6f}%).')
    print_results(report)


if __name__ == '__main__':
    main()
