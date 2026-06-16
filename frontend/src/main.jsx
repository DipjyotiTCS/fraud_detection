import React, { useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const DEFAULT_API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const DEFAULT_SETTINGS = {
  apiBaseUrl: DEFAULT_API_BASE_URL,
  artifactDir: '/workspace/shared/fraud_detection/artifacts/xgboost',
  modelVersion: 'v1',
  baseModelPath: 'mistralai/Mistral-7B-Instruct-v0.3',
  adapterPath: '/workspace/shared/mistral_dpo_v3',
  use4bit: false,
  torchDtype: 'bfloat16',
  maxNewTokens: 2048,
  temperature: 0.0,
  topP: 0.9,
};

const PIPELINE_STEPS = [
  {
    id: 'transaction',
    title: 'Transaction record ready',
    description: 'Generate a new transaction using the same JSON structure used by the notebook.',
  },
  {
    id: 'classification',
    title: 'GNN + XGBoost classification completed',
    description: 'Call /api/xgboost/score with the generated transaction.',
  },
  {
    id: 'llmPayload',
    title: 'LLM inference request prepared',
    description: 'Build the notebook-aligned payload with transaction and xgboost_score_response.',
  },
  {
    id: 'llmInference',
    title: 'Local LLM inference completed',
    description: 'Call /api/llm/infer-fraud-report using the hosted base model + LoRA.',
  },
  {
    id: 'cleanReport',
    title: 'LLM JSON cleaned and parsed',
    description: 'Use parsed_report if available, otherwise clean raw_response into valid JSON.',
  },
  {
    id: 'visualReport',
    title: 'Visual report rendered',
    description: 'Render the final report as investigation-friendly cards and evidence sections.',
  },
];

function randomId(prefix, min = 100, max = 999) {
  return `${prefix}-${Math.floor(min + Math.random() * (max - min + 1))}`;
}

function randomInt(min, max) {
  return Math.floor(min + Math.random() * (max - min + 1));
}

function randomFloat(min, max, digits = 2) {
  return Number((min + Math.random() * (max - min)).toFixed(digits));
}

function pick(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function generateTransactionRecord() {
  const highRisk = Math.random() > 0.35;
  const amountInr = highRisk ? randomInt(420000, 1200000) : randomInt(2500, 180000);
  const amountMean90d = highRisk ? randomInt(15000, 55000) : randomInt(20000, 120000);
  const amountStd90d = highRisk ? randomInt(3500, 12000) : randomInt(8000, 40000);
  const transactionDate = new Date(Date.now() - randomInt(0, 90) * 24 * 60 * 60 * 1000);
  const watchlistHit = highRisk ? Math.random() > 0.82 : false;

  return {
    transaction_id: randomId('test-txn', 100, 999),
    account_id_hash: randomId('test-account', 1, 250),
    beneficiary_id_hash: randomId('test-beneficiary', 100, 350),
    device_fingerprint_hash: randomId('test-device', 1, 80),
    transaction_dt: transactionDate.toISOString().replace('.000Z', 'Z'),
    amount_inr: amountInr,
    amount_usd_equiv: Math.round(amountInr / 83.7),
    channel: pick(['UPI', 'IMPS', 'NEFT', 'CARD_CNP', 'NETBANKING', 'WALLET']),
    merchant_category_code: pick(['6012', '4829', '6051', '5734', '5812', '5999', '6211']),
    is_international: highRisk ? Math.random() > 0.65 : Math.random() > 0.9,
    currency: 'INR',
    is_new_beneficiary: highRisk ? Math.random() > 0.2 : Math.random() > 0.75,
    days_since_account_open: highRisk ? randomInt(5, 90) : randomInt(120, 2400),
    velocity_1h: highRisk ? randomInt(4, 12) : randomInt(0, 3),
    velocity_6h: highRisk ? randomInt(9, 28) : randomInt(0, 7),
    velocity_24h: highRisk ? randomInt(22, 58) : randomInt(1, 12),
    velocity_7d: highRisk ? randomInt(50, 130) : randomInt(4, 35),
    amount_sum_1h: highRisk ? randomInt(amountInr, amountInr * 4) : randomInt(amountInr, amountInr * 2),
    amount_sum_24h: highRisk ? randomInt(amountInr * 3, amountInr * 10) : randomInt(amountInr, amountInr * 4),
    distinct_beneficiaries_6h: highRisk ? randomInt(3, 9) : randomInt(1, 3),
    distinct_beneficiaries_24h: highRisk ? randomInt(5, 15) : randomInt(1, 5),
    ip_reputation_score: highRisk ? randomFloat(0.65, 0.98) : randomFloat(0.03, 0.45),
    is_tor_vpn_proxy: highRisk ? Math.random() > 0.45 : Math.random() > 0.92,
    geo_distance_km: highRisk ? randomInt(850, 6200) : randomInt(5, 450),
    geo_country_mismatch: highRisk ? Math.random() > 0.55 : Math.random() > 0.96,
    watchlist_hit: watchlistHit,
    watchlist_category: watchlistHit ? pick(['SANCTIONS', 'PEP', 'ADVERSE_MEDIA']) : 'NONE',
    failed_auth_count_7d: highRisk ? randomInt(2, 10) : randomInt(0, 2),
    amount_mean_90d: amountMean90d,
    amount_std_90d: amountStd90d,
    typical_velocity_daily: highRisk ? randomInt(2, 8) : randomInt(2, 16),
    is_dormant_account: highRisk ? Math.random() > 0.72 : Math.random() > 0.92,
    distinct_devices_30d: highRisk ? randomInt(3, 10) : randomInt(1, 4),
    behavioural_drift_score: highRisk ? randomFloat(0.58, 0.96) : randomFloat(0.02, 0.38),
    peer_group_id: `PG-${randomInt(100, 899)}`,
    peer_amount_percentile: highRisk ? randomInt(88, 100) : randomInt(15, 80),
    shared_device_accounts: highRisk ? randomInt(4, 18) : randomInt(0, 3),
    shared_ip_accounts: highRisk ? randomInt(6, 28) : randomInt(0, 5),
    beneficiary_sar_count: highRisk ? randomInt(0, 3) : 0,
    second_hop_sar_count: highRisk ? randomInt(1, 6) : randomInt(0, 1),
    mule_network_probability: highRisk ? randomFloat(0.55, 0.94) : randomFloat(0.01, 0.25),
    synthetic_identity_score: highRisk ? randomFloat(0.30, 0.78) : randomFloat(0.01, 0.20),
    rapid_fan_out_flag: highRisk ? Math.random() > 0.28 : false,
    originator_graph_centrality: highRisk ? randomFloat(0.35, 0.88) : randomFloat(0.01, 0.26),
  };
}

function initialStepState() {
  return PIPELINE_STEPS.reduce((acc, step) => {
    acc[step.id] = { status: 'pending', detail: '' };
    return acc;
  }, {});
}

async function postJson(baseUrl, path, payload, timeoutMs = 1_800_000) {
  const controller = new AbortController();
  const timerId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${baseUrl}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    const text = await response.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch {
      json = { raw_text: text };
    }

    if (!response.ok) {
      const detail = json?.detail?.message || json?.detail || json?.message || text || response.statusText;
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }

    return json;
  } finally {
    clearTimeout(timerId);
  }
}

function cleanAndParseJson(rawText) {
  if (!rawText || typeof rawText !== 'string') {
    throw new Error('No raw_response text was available to parse.');
  }

  let text = rawText.trim();
  text = text.replace(/^```json\s*/i, '').replace(/^```\s*/i, '').replace(/\s*```$/i, '');

  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');

  if (start === -1) {
    throw new Error('No JSON object found in raw_response.');
  }

  let jsonText = end === -1 ? text.slice(start) : text.slice(start, end + 1);

  const openCount = (jsonText.match(/{/g) || []).length;
  const closeCount = (jsonText.match(/}/g) || []).length;
  if (openCount > closeCount) {
    jsonText += '}'.repeat(openCount - closeCount);
  }

  jsonText = jsonText.replace(/,\s*}/g, '}').replace(/,\s*]/g, ']');
  return JSON.parse(jsonText);
}

function normalizeLlmReport(llmResponse) {
  if (llmResponse?.parsed_report && typeof llmResponse.parsed_report === 'object') {
    return {
      report: llmResponse.parsed_report,
      parseSource: 'backend_parsed_report',
      parseError: null,
    };
  }

  try {
    return {
      report: cleanAndParseJson(llmResponse?.raw_response || llmResponse?.report || ''),
      parseSource: 'frontend_cleaned_raw_response',
      parseError: null,
    };
  } catch (error) {
    return {
      report: null,
      parseSource: 'failed',
      parseError: error.message,
    };
  }
}

function StepTracker({ stepState }) {
  return (
    <div className="panel step-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Pipeline</p>
          <h2>Frontend Orchestration Steps</h2>
        </div>
      </div>

      <div className="steps">
        {PIPELINE_STEPS.map((step, index) => {
          const state = stepState[step.id] || { status: 'pending', detail: '' };
          return (
            <div className={`step ${state.status}`} key={step.id}>
              <div className="step-marker">
                {state.status === 'done' ? '✓' : state.status === 'active' ? '…' : state.status === 'error' ? '!' : index + 1}
              </div>
              <div className="step-body">
                <div className="step-title">{step.title}</div>
                <div className="step-description">{step.description}</div>
                {state.detail ? <div className="step-detail">{state.detail}</div> : null}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SettingsPanel({ settings, setSettings }) {
  function update(key, value) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  return (
    <div className="panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Configuration</p>
          <h2>Backend API Settings</h2>
        </div>
      </div>

      <div className="form-grid">
        <label>
          Backend URL
          <input value={settings.apiBaseUrl} onChange={(e) => update('apiBaseUrl', e.target.value)} />
        </label>
        <label>
          XGBoost Artifact Dir
          <input value={settings.artifactDir} onChange={(e) => update('artifactDir', e.target.value)} />
        </label>
        <label>
          XGBoost Model Version
          <input value={settings.modelVersion} onChange={(e) => update('modelVersion', e.target.value)} />
        </label>
        <label>
          Base Model
          <input value={settings.baseModelPath} onChange={(e) => update('baseModelPath', e.target.value)} />
        </label>
        <label>
          LoRA Adapter Path
          <input value={settings.adapterPath} onChange={(e) => update('adapterPath', e.target.value)} />
        </label>
        <label>
          Torch dtype
          <select value={settings.torchDtype} onChange={(e) => update('torchDtype', e.target.value)}>
            <option value="bfloat16">bfloat16</option>
            <option value="float16">float16</option>
            <option value="float32">float32</option>
            <option value="auto">auto</option>
          </select>
        </label>
      </div>
    </div>
  );
}

function TransactionEditor({ transactionText, setTransactionText, onGenerate }) {
  return (
    <div className="panel transaction-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Input</p>
          <h2>Transaction Record</h2>
        </div>
        <button className="secondary-btn" onClick={onGenerate}>Generate New Transaction</button>
      </div>
      <p className="helper-text">
        This object follows the same structure as the notebook <code>payload["records"][0]</code>. You can edit it before running the report.
      </p>
      <textarea
        className="json-editor"
        value={transactionText}
        onChange={(e) => setTransactionText(e.target.value)}
        spellCheck="false"
      />
    </div>
  );
}

function SummaryMetric({ label, value, tone }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${tone || ''}`}>{value ?? 'N/A'}</div>
    </div>
  );
}

function EvidenceList({ items }) {
  if (!Array.isArray(items) || items.length === 0) {
    return <p className="muted">No evidence available.</p>;
  }

  return (
    <ul className="evidence-list">
      {items.map((item, index) => <li key={`${item}-${index}`}>{String(item)}</li>)}
    </ul>
  );
}

function AlternativeCards({ items }) {
  if (!Array.isArray(items) || items.length === 0) {
    return <p className="muted">No alternatives available.</p>;
  }

  return (
    <div className="mini-card-grid">
      {items.map((item, index) => (
        <div className="mini-card" key={index}>
          <strong>{item.code || 'N/A'} — {item.name || 'N/A'}</strong>
          <p>{item.reason_rejected || 'No reason supplied.'}</p>
        </div>
      ))}
    </div>
  );
}

function FeatureContributionRows({ items }) {
  if (!Array.isArray(items) || items.length === 0) {
    return <p className="muted">No feature contribution evidence available.</p>;
  }

  return (
    <div className="feature-list">
      {items.map((line, index) => {
        const text = String(line);
        const feature = text.includes(':') ? text.split(':')[0] : `Feature ${index + 1}`;
        const pctMatch = text.match(/contribution_percentage=([0-9.]+)%?/);
        const pct = pctMatch ? Number(pctMatch[1]) : 0;
        const width = Math.max(0, Math.min(pct, 100));
        return (
          <div className="feature-row" key={index}>
            <div>
              <strong>{feature}</strong>
              <span>{text}</span>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${width}%` }} />
            </div>
            <em>{pctMatch ? `${pct}%` : 'N/A'}</em>
          </div>
        );
      })}
    </div>
  );
}

function VisualReport({ report, xgboostScore, transaction }) {
  if (!report) return null;

  const typology = report.fraud_typology || {};
  const classification = report.fraud_classification || {};
  const rationale = report.rationale || {};
  const sar = report.sar_narrative || {};
  const nextAction = report.next_best_action || {};

  const riskTier = String(classification.risk_tier || 'UNKNOWN').toLowerCase();
  const fraudTone = classification.is_fraud ? 'danger-text' : 'success-text';

  return (
    <div className="report-shell">
      <div className="report-hero">
        <div>
          <p className="eyebrow light">Visual Report</p>
          <h1>Fraud Investigation Report</h1>
          <p>Transaction {transaction?.transaction_id || 'N/A'} · {transaction?.channel || 'N/A'} · INR {transaction?.amount_inr || 'N/A'}</p>
        </div>
        <div className={`tier-pill ${riskTier}`}>{classification.risk_tier || 'UNKNOWN'}</div>
      </div>

      <div className="metric-grid">
        <SummaryMetric label="Classification" value={classification.classification || 'UNKNOWN'} tone={fraudTone} />
        <SummaryMetric label="Risk Score" value={classification.risk_score ?? xgboostScore?.risk_score} />
        <SummaryMetric label="Probability" value={`${classification.probability_percentage ?? xgboostScore?.probability_percentage ?? 'N/A'}%`} />
        <SummaryMetric label="Typology" value={`${typology.code || 'N/A'} ${typology.name ? `· ${typology.name}` : ''}`} />
      </div>

      <section className="report-section">
        <h2>Risk Rationale</h2>
        <div className="soft-box">{rationale.summary || 'No rationale summary returned.'}</div>
        <div className="two-column">
          <div>
            <h3>Model Evidence</h3>
            <EvidenceList items={rationale.model_evidence} />
          </div>
          <div>
            <h3>GNN Evidence</h3>
            <EvidenceList items={rationale.gnn_evidence} />
          </div>
        </div>
      </section>

      <section className="report-section">
        <h2>Fraud Typology</h2>
        <div className="soft-box">
          <strong>{typology.code || 'N/A'} — {typology.name || 'N/A'}</strong>
          <br />Confidence: {typology.confidence ?? 'N/A'}
        </div>
        <h3>Supporting Evidence</h3>
        <EvidenceList items={typology.supporting_evidence} />
        <h3>Alternative Typologies Considered</h3>
        <AlternativeCards items={typology.alternative_typologies_considered} />
      </section>

      <section className="report-section">
        <h2>Feature Contribution Evidence</h2>
        <FeatureContributionRows items={rationale.feature_contribution_evidence} />
      </section>

      <section className="report-section">
        <h2>SAR Narrative</h2>
        <div className="soft-box">
          <strong>Should File SAR:</strong> {String(sar.should_file_sar ?? 'N/A')}
          <br /><br />
          {sar.narrative || 'No SAR narrative returned.'}
        </div>
      </section>

      <section className="report-section">
        <h2>Next Best Action</h2>
        <div className="soft-box"><strong>{nextAction.recommendation || 'N/A'}</strong></div>
        <h3>Supporting Evidence</h3>
        <EvidenceList items={nextAction.supporting_evidence} />
        <h3>Alternative Actions Considered</h3>
        <AlternativeCards items={nextAction.alternative_actions_considered} />
      </section>
    </div>
  );
}

function DebugPanel({ xgboostPayload, xgboostResponse, llmPayload, llmResponse, finalReport }) {
  return (
    <details className="panel debug-panel">
      <summary>Debug JSON</summary>
      <div className="debug-grid">
        <JsonBlock title="XGBoost Request" value={xgboostPayload} />
        <JsonBlock title="XGBoost Response" value={xgboostResponse} />
        <JsonBlock title="LLM Request" value={llmPayload} />
        <JsonBlock title="LLM Response" value={llmResponse} />
        <JsonBlock title="Final Parsed Report" value={finalReport} />
      </div>
    </details>
  );
}

function JsonBlock({ title, value }) {
  return (
    <div>
      <h3>{title}</h3>
      <pre>{value ? JSON.stringify(value, null, 2) : 'Not available yet.'}</pre>
    </div>
  );
}

function App() {
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const initialTransaction = useMemo(() => generateTransactionRecord(), []);
  const [transaction, setTransaction] = useState(initialTransaction);
  const [transactionText, setTransactionText] = useState(() => JSON.stringify(initialTransaction, null, 2));
  const [stepState, setStepState] = useState(initialStepState);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [xgboostPayload, setXgboostPayload] = useState(null);
  const [xgboostResponse, setXgboostResponse] = useState(null);
  const [xgboostScore, setXgboostScore] = useState(null);
  const [llmPayload, setLlmPayload] = useState(null);
  const [llmResponse, setLlmResponse] = useState(null);
  const [finalReport, setFinalReport] = useState(null);
  const [parseSource, setParseSource] = useState('');

  const completedCount = useMemo(
    () => PIPELINE_STEPS.filter((step) => stepState[step.id]?.status === 'done').length,
    [stepState]
  );

  function setStep(id, status, detail = '') {
    setStepState((current) => ({
      ...current,
      [id]: { status, detail },
    }));
  }

  function handleGenerateTransaction() {
    const nextTransaction = generateTransactionRecord();
    setTransaction(nextTransaction);
    setTransactionText(JSON.stringify(nextTransaction, null, 2));
    setFinalReport(null);
    setXgboostPayload(null);
    setXgboostResponse(null);
    setXgboostScore(null);
    setLlmPayload(null);
    setLlmResponse(null);
    setError('');
    const nextSteps = initialStepState();
    nextSteps.transaction = { status: 'done', detail: `Generated ${nextTransaction.transaction_id}` };
    setStepState(nextSteps);
  }

  async function runReportPipeline() {
    let currentStepId = null;
    const markStep = (id, status, detail = '') => {
      currentStepId = id;
      setStep(id, status, detail);
    };

    setRunning(true);
    setError('');
    setFinalReport(null);
    setXgboostPayload(null);
    setXgboostResponse(null);
    setXgboostScore(null);
    setLlmPayload(null);
    setLlmResponse(null);
    setParseSource('');
    setStepState(initialStepState());

    try {
      markStep('transaction', 'active', 'Validating transaction JSON.');
      const parsedTransaction = JSON.parse(transactionText);
      setTransaction(parsedTransaction);
      markStep('transaction', 'done', `Ready: ${parsedTransaction.transaction_id || 'transaction_id missing'}`);

      markStep('classification', 'active', 'Calling /api/xgboost/score.');
      const scorePayload = {
        artifact_dir: settings.artifactDir,
        model_version: settings.modelVersion,
        records: [parsedTransaction],
      };
      setXgboostPayload(scorePayload);

      const scoreResponse = await postJson(settings.apiBaseUrl, '/api/xgboost/score', scorePayload, 300_000);
      setXgboostResponse(scoreResponse);

      const score = scoreResponse?.scores?.[0];
      if (!score) {
        throw new Error('The XGBoost response did not contain scores[0].');
      }
      setXgboostScore(score);
      markStep('classification', 'done', `Risk score ${score.risk_score ?? 'N/A'}, probability ${score.probability_percentage ?? score.probability ?? 'N/A'}.`);

      markStep('llmPayload', 'active', 'Preparing notebook-aligned LLM inference payload.');
      const inferencePayload = {
        transaction: parsedTransaction,
        xgboost_score_response: score,
        base_model_path: settings.baseModelPath,
        adapter_path: settings.adapterPath,
        use_4bit: settings.use4bit,
        torch_dtype: settings.torchDtype,
        max_new_tokens: Number(settings.maxNewTokens),
        temperature: Number(settings.temperature),
        top_p: Number(settings.topP),
      };
      setLlmPayload(inferencePayload);
      markStep('llmPayload', 'done', 'Payload prepared using transaction + xgboost_score_response.');

      markStep('llmInference', 'active', 'Calling /api/llm/infer-fraud-report. First call may load the model.');
      const inferenceResponse = await postJson(settings.apiBaseUrl, '/api/llm/infer-fraud-report', inferencePayload, 1_800_000);
      setLlmResponse(inferenceResponse);
      markStep('llmInference', 'done', 'LLM response received.');

      markStep('cleanReport', 'active', 'Normalizing parsed_report/raw_response into final report JSON.');
      const normalized = normalizeLlmReport(inferenceResponse);
      if (!normalized.report) {
        throw new Error(`Could not parse LLM report JSON. ${normalized.parseError || ''}`);
      }
      setFinalReport(normalized.report);
      setParseSource(normalized.parseSource);
      markStep('cleanReport', 'done', `Report JSON ready from ${normalized.parseSource}.`);

      markStep('visualReport', 'active', 'Rendering visual report.');
      markStep('visualReport', 'done', 'Visual report rendered successfully.');
    } catch (err) {
      const message = err?.message || String(err);
      setError(message);
      if (currentStepId) {
        setStep(currentStepId, 'error', message);
      }
    } finally {
      setRunning(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">FraudSentinel</p>
          <h1>Local LLM Fraud Report Console</h1>
          <p className="topbar-subtitle">
            React orchestrates the workflow: generate transaction → score with GNN/XGBoost → call LoRA-backed LLM → clean JSON → render visual report.
          </p>
        </div>
        <div className="progress-card">
          <span>{completedCount}/{PIPELINE_STEPS.length}</span>
          <small>steps complete</small>
        </div>
      </header>

      <div className="action-row">
        <button className="primary-btn" onClick={runReportPipeline} disabled={running}>
          {running ? 'Running Report Pipeline...' : 'Trigger Report Generation'}
        </button>
        <button className="secondary-btn" onClick={handleGenerateTransaction} disabled={running}>
          Generate Transaction Record
        </button>
        {parseSource ? <span className="parse-source">JSON source: {parseSource}</span> : null}
      </div>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="layout-grid">
        <div className="left-stack">
          <SettingsPanel settings={settings} setSettings={setSettings} />
          <TransactionEditor
            transactionText={transactionText}
            setTransactionText={setTransactionText}
            onGenerate={handleGenerateTransaction}
          />
        </div>
        <StepTracker stepState={stepState} />
      </div>

      <VisualReport report={finalReport} xgboostScore={xgboostScore} transaction={transaction} />

      <DebugPanel
        xgboostPayload={xgboostPayload}
        xgboostResponse={xgboostResponse}
        llmPayload={llmPayload}
        llmResponse={llmResponse}
        finalReport={finalReport}
      />
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
