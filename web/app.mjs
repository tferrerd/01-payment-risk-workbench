export function decisionFor(score, amount, margin = .003, loss = 1) {
  return score > margin / (margin + loss) && amount > 0 ? 'cancel' : 'approve';
}

export function profitAtCutoff(rows, cutoff, margin, loss) {
  return rows.reduce((sum, row) => {
    const { Amount: amount } = row.features;
    return sum + (row.prediction.fraud_probability > cutoff && amount > 0 ? 0 : amount * (row.label === 1 ? -loss : margin));
  }, 0);
}

export function summarize(rows, margin, loss) {
  const result = { approved: 0, canceled: 0, fraudsCanceled: 0, fraudAmount: 0, legitCanceled: 0, marginLost: 0, profit: 0, baseline: 0 };
  for (const row of rows) {
    const amount = row.features.Amount;
    const payoff = amount * (row.label === 1 ? -loss : margin);
    result.baseline += payoff;
    if (decisionFor(row.prediction.fraud_probability, amount, margin, loss) === 'cancel') {
      result.canceled++;
      if (row.label === 1) { result.fraudsCanceled++; result.fraudAmount += amount; }
      else { result.legitCanceled++; result.marginLost += amount * margin; }
    } else { result.approved++; result.profit += payoff; }
  }
  return result;
}

export function exportCsv(rows, margin, loss) {
  const fields = ['payment_id', 'source_row_id', 'published_label', 'Amount', 'Time', 'fraud_score', 'api_decision', 'simulated_decision', 'legitimate_margin_fraction', 'fraud_loss_fraction', 'model_id'];
  const values = rows.map(row => [row.payment_id, row.source_row_id, row.label, row.features.Amount, row.features.Time,
    row.prediction.fraud_probability, row.prediction.decision, decisionFor(row.prediction.fraud_probability, row.features.Amount, margin, loss), margin, loss, row.prediction.model_id]);
  return [fields, ...values].map(row => row.map(value => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\n') + '\n';
}

if (typeof document !== 'undefined') {
  const $ = id => document.getElementById(id);
  const state = { rows: [], margin: .003, loss: 1, page: 0, selected: null, explanationRequest: 0 };
  const number = value => new Intl.NumberFormat('en', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value);
  const signed = value => `${value >= 0 ? '+' : '−'}${number(Math.abs(value))}`;
  const pct = value => `${(value * 100).toFixed(value < .0001 ? 4 : 2)}%`;
  const escape = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
  const decision = row => decisionFor(row.prediction.fraud_probability, row.features.Amount, state.margin, state.loss);
  const badge = value => `<span class="badge ${value}">${value === 'cancel' ? 'Cancel' : 'Approve'}</span>`;

  async function json(url, body) {
    const response = await fetch(url, body === undefined ? undefined : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!response.ok) throw new Error(`The model service returned ${response.status}. Please reload to try again.`);
    return response.json();
  }

  function filtered() {
    const query = $('search').value.trim().toLowerCase();
    return state.rows.filter(row => {
      const action = decision(row), filter = $('filter').value;
      return row.payment_id.toLowerCase().includes(query) && (filter === 'all' || filter === action ||
        (filter === 'fraud' && row.label === 1) || (filter === 'legitimate_cancel' && row.label === 0 && action === 'cancel'));
    }).sort((a, b) => $('sort').value === 'amount' ? b.features.Amount - a.features.Amount :
      $('sort').value === 'time' ? a.features.Time - b.features.Time : b.prediction.fraud_probability - a.prediction.fraud_probability);
  }

  function renderTable() {
    const rows = filtered();
    const pages = Math.max(1, Math.ceil(rows.length / 10));
    state.page = Math.min(state.page, pages - 1);
    $('transactions').innerHTML = rows.slice(state.page * 10, (state.page + 1) * 10).map(row => {
      const score = row.prediction.fraud_probability, action = decision(row);
      return `<tr data-payment-id="${escape(row.payment_id)}" class="${state.selected === row.payment_id ? 'selected' : ''}"><td><button class="payment-button" data-payment-id="${escape(row.payment_id)}">${escape(row.payment_id)}</button></td><td>${number(row.features.Amount)}</td><td><span class="risk-cell"><i class="risk-indicator ${score > state.margin / (state.margin + state.loss) ? 'high' : 'low'}"></i>${pct(score)}</span></td><td class="${row.label ? 'label-fraud' : 'label-legitimate'}">${row.label ? 'Fraud' : 'Legitimate'}</td><td>${badge(action)}</td></tr>`;
    }).join('') || '<tr><td colspan="5" class="empty">No payments match these filters.</td></tr>';
    $('row-count').textContent = rows.length;
    $('page-info').textContent = rows.length ? `${state.page * 10 + 1}–${Math.min((state.page + 1) * 10, rows.length)} of ${rows.length} payments` : '0 payments';
    $('previous').disabled = state.page === 0;
    $('next').disabled = state.page >= pages - 1;
  }

  function renderDetail() {
    const row = state.rows.find(row => row.payment_id === state.selected);
    if (!row) return;
    $('detail-empty').hidden = true;
    $('detail').hidden = false;
    $('selected-id').textContent = row.payment_id;
    const action = decision(row), score = row.prediction.fraud_probability;
    $('selected-decision').textContent = action === 'cancel' ? 'Cancel' : 'Approve';
    $('selected-decision').className = `badge ${action}`;
    $('selected-score').textContent = pct(score);
    $('risk-fill').style.width = `${Math.max(.5, score * 100)}%`;
    $('risk-fill').style.background = score > state.margin / (state.margin + state.loss) ? 'var(--red)' : 'var(--teal)';
    $('selected-amount').textContent = number(row.features.Amount);
    $('selected-time').textContent = (row.features.Time / 3600).toFixed(2);
    $('selected-label').textContent = row.label === 1 ? 'Fraud' : 'Legitimate';
    const profit = row.features.Amount * (state.margin - (state.margin + state.loss) * score);
    $('selected-profit').textContent = signed(profit);
    $('selected-profit').classList.toggle('negative-value', profit < 0);
  }

  async function selectPayment(id) {
    state.selected = id;
    renderTable(); renderDetail();
    const request = ++state.explanationRequest;
    $('contributions').textContent = 'Calculating model contributions…';
    $('explanation-detail').textContent = '';
    try {
      const row = state.rows.find(row => row.payment_id === id);
      const result = await json('/explain', row.features);
      if (request !== state.explanationRequest) return;
      const ordered = [...result.contributions].sort((a, b) => Math.abs(b.log_odds) - Math.abs(a.log_odds));
      const shown = [...ordered.slice(0, 6), { feature: 'Other', log_odds: ordered.slice(6).reduce((sum, item) => sum + item.log_odds, 0) }];
      const max = Math.max(.001, ...shown.map(item => Math.abs(item.log_odds)));
      $('contributions').innerHTML = shown.map(item => {
        const width = Math.abs(item.log_odds) / max * 47;
        return `<div class="contribution"><span>${escape(item.feature)}</span><div class="contribution-track"><span class="contribution-bar ${item.log_odds < 0 ? 'negative' : ''}" style="width:${width}%;left:${item.log_odds < 0 ? 50 - width : 50}%"></span></div><span>${item.log_odds >= 0 ? '+' : '−'}${Math.abs(item.log_odds).toFixed(2)}</span></div>`;
      }).join('');
      $('explanation-detail').textContent = `Base ${result.base_log_odds.toFixed(3)} + all contributions = ${result.raw_log_odds.toFixed(3)} log-odds. “Other” sums the remaining 24 features.`;
    } catch (error) { if (request === state.explanationRequest) $('contributions').textContent = `Explanation unavailable. ${error.message}`; }
  }

  function renderChart() {
    const points = Array.from({ length: 101 }, (_, index) => {
      const threshold = Math.pow(10, -5 + index / 20);
      return [threshold, profitAtCutoff(state.rows, threshold, state.margin, state.loss)];
    });
    const cutoff = state.margin / (state.margin + state.loss);
    const current = profitAtCutoff(state.rows, cutoff, state.margin, state.loss);
    const baseline = profitAtCutoff(state.rows, 1, state.margin, state.loss);
    const values = [...points.map(point => point[1]), current, baseline, 0];
    const min = Math.min(...values), max = Math.max(...values), pad = Math.max(1, (max - min) * .14);
    const x = threshold => 78 + (Math.log10(Math.max(.00001, threshold)) + 5) / 5 * 615;
    const y = profit => 220 - (profit - min + pad) / (max - min + 2 * pad) * 195;
    const labels = [.00001, .0001, .001, .01, .1, 1].map(value => `<text x="${x(value)}" y="248" text-anchor="middle" fill="#93a0b0" font-size="10">${value * 100}%</text>`).join('');
    const grid = Array.from({ length: 4 }, (_, index) => {
      const value = min + (max - min || 1) * index / 3;
      return `<path d="M78 ${y(value)}H693" stroke="#edf1f5"/><text x="65" y="${y(value) + 4}" text-anchor="end" fill="#93a0b0" font-size="10">${new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value)}</text>`;
    }).join('');
    const path = points.map(([threshold, profit], index) => `${index ? 'L' : 'M'}${x(threshold).toFixed(2)},${y(profit).toFixed(2)}`).join(' ');
    $('profit-chart').innerHTML = `<title>Simulated profit on the curated labeled sample; current cutoff ${(cutoff * 100).toFixed(4)} percent</title>${grid}${labels}<path d="M78 ${y(baseline)}H693" stroke="#aab5c3" stroke-dasharray="5 5"/><path d="${path}" fill="none" stroke="#087e71" stroke-width="2.5" stroke-linejoin="round"/><path d="M${x(cutoff)} 18V224" stroke="#3c61da" stroke-dasharray="4 4"/><circle cx="${x(cutoff)}" cy="${y(current)}" r="5" fill="#3c61da" stroke="white" stroke-width="2"/><text x="${Math.min(x(cutoff) + 8, 610)}" y="15" font-size="9" fill="#3c61da">Break-even cutoff</text>`;
    $('chart-policy').textContent = `Current policy: ${signed(current)}`;
  }

  function renderPolicy() {
    const summary = summarize(state.rows, state.margin, state.loss);
    $('approved').textContent = summary.approved;
    $('approval-rate').textContent = `${(100 * summary.approved / state.rows.length).toFixed(1)}% of ${state.rows.length} sample payments`;
    $('fraud-caught').textContent = number(summary.fraudAmount);
    $('fraud-count').textContent = `${summary.fraudsCanceled} of ${state.rows.filter(row => row.label === 1).length} labeled frauds canceled`;
    $('legit-canceled').textContent = summary.legitCanceled;
    $('margin-lost').textContent = `${number(summary.marginLost)} in assumed margin forgone`;
    $('sample-profit').textContent = signed(summary.profit);
    $('sample-profit').classList.toggle('negative-value', summary.profit < 0);
    $('profit-delta').textContent = `${signed(summary.profit - summary.baseline)} versus approve-all`;
    $('margin-value').textContent = `${(state.margin * 100).toFixed(2)}%`;
    $('fraud-loss-value').textContent = `${(state.loss * 100).toFixed(0)}%`;
    $('threshold').textContent = `${(state.margin / (state.margin + state.loss) * 100).toFixed(4)}%`;
    const changed = Math.abs(state.margin - .003) > 1e-12 || Math.abs(state.loss - 1) > 1e-12;
    $('policy-state').textContent = changed ? 'What-if simulation' : 'Default policy';
    $('policy-state').classList.toggle('changed', changed);
    renderTable(); renderDetail(); renderChart();
  }

  for (const id of ['search', 'filter', 'sort']) $(id).addEventListener('input', () => { state.page = 0; renderTable(); });
  $('previous').addEventListener('click', () => { state.page--; renderTable(); });
  $('next').addEventListener('click', () => { state.page++; renderTable(); });
  $('transactions').addEventListener('click', event => { const target = event.target.closest('[data-payment-id]'); if (target) selectPayment(target.dataset.paymentId); });
  for (const id of ['margin', 'fraud-loss']) $(id).addEventListener('input', () => {
    state.margin = Number($('margin').value) / 100;
    state.loss = Number($('fraud-loss').value) / 100;
    if (state.rows.length) renderPolicy();
  });
  $('reset').addEventListener('click', () => {
    $('margin').value = .3; $('fraud-loss').value = 100;
    state.margin = .003; state.loss = 1;
    if (state.rows.length) renderPolicy();
  });
  $('export').addEventListener('click', () => {
    const url = URL.createObjectURL(new Blob([exportCsv(filtered(), state.margin, state.loss)], { type: 'text/csv;charset=utf-8' }));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'payment-risk-sample-simulation.csv'; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  async function load() {
    try {
      const demo = await json('/demo');
      const scored = await json('/predict', { transactions: demo.rows.map(row => row.features) });
      if (scored.predictions.length !== demo.rows.length) throw new Error('The model returned an incomplete batch.');
      state.rows = demo.rows.map((row, index) => ({ ...row, prediction: scored.predictions[index] }));
      $('sample-description').textContent = `${state.rows.length} curated selection examples · ${state.rows.filter(row => row.label === 1).length} labeled frauds · enriched demo sample`;
      $('connection-status').textContent = 'LightGBM · model connected';
      $('export').disabled = false;
      renderPolicy();
      const example = state.rows.find(row => row.label === 1 && decision(row) === 'cancel') || state.rows[0];
      await selectPayment(example.payment_id);
    } catch (error) {
      $('error').hidden = false;
      $('error').textContent = `${error.message} The demo needs its live inference service to score payments.`;
      $('connection-status').textContent = 'Model unavailable';
      $('transactions').innerHTML = '<tr><td colspan="5" class="empty">Unable to load payments. Reload to retry.</td></tr>';
    }
  }
  load();
}
