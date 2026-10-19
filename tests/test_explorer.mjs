import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { decisionFor, summarize, profitAtCutoff, exportCsv } from '../web/app.mjs';

test('browser economics match API ties, zero amounts, and monetary payoffs', () => {
  const cutoff = .003 / 1.003;
  assert.equal(decisionFor(cutoff, 100), 'approve');
  assert.equal(decisionFor(cutoff + 1e-9, 100), 'cancel');
  assert.equal(decisionFor(1, 0), 'approve');
  const rows = [
    { payment_id: 'one', source_row_id: 1, label: 0, features: { Amount: 100, Time: 0 }, prediction: { fraud_probability: 0, decision: 'approve', model_id: 'test' } },
    { payment_id: 'two', source_row_id: 2, label: 1, features: { Amount: 100, Time: 1 }, prediction: { fraud_probability: .01, decision: 'cancel', model_id: 'test' } },
  ];
  const base = summarize(rows, .003, 1);
  assert.equal(base.profit, .3);
  assert.equal(base.baseline, -99.7);
  assert.equal(base.fraudAmount, 100);
  assert.equal(base.approved, 1);
  assert.equal(base.canceled, 1);
  assert.equal(summarize(rows, .25, 1).profit, -75);
  assert.equal(profitAtCutoff(rows, cutoff, .003, 1), base.profit);
  assert.equal(profitAtCutoff(rows, 1, .003, 1), base.baseline);
  const csv = exportCsv(rows, .25, 1);
  assert.match(csv, /api_decision/);
  assert.match(csv, /"cancel","approve","0.25"/);
  assert.equal(rows[1].prediction.decision, 'cancel');
});

test('saved source-derived example decisions match the packaged policy', () => {
  const inputs = JSON.parse(readFileSync(new URL('../examples/predict-request.json', import.meta.url)));
  const outputs = JSON.parse(readFileSync(new URL('../examples/predict-expected.json', import.meta.url)));
  inputs.transactions.forEach((row, index) => assert.equal(decisionFor(outputs[index].fraud_probability, row.Amount), outputs[index].decision));
});
