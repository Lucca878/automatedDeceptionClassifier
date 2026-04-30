import './style.css';
import Papa from 'papaparse';
import Chart from 'chart.js/auto';

const BACKEND_URL = 'http://localhost:8000';

type DeceptionLabel = 'deceptive' | 'truthful';

interface Prediction {
  label: DeceptionLabel;
  confidence: number;
  rawLabel: string;
}

interface Metrics {
  accuracy: number;
  precisionTruthful: number;
  recallTruthful: number;
  f1Truthful: number;
  precisionDeceptive: number;
  recallDeceptive: number;
  f1Deceptive: number;
  f1Macro: number;
  support: number;
  confusion: {
    truthfulTrue: number;
    truthfulFalse: number;
    deceptiveTrue: number;
    deceptiveFalse: number;
  };
}

type CsvRow = Record<string, string>;

// 'auto'        → read id2label from config.json inside the uploaded model
// '0t1d'        → LABEL_0 = truthful, LABEL_1 = deceptive  (training-pipeline semantics)
// '0d1t'        → LABEL_0 = deceptive, LABEL_1 = truthful
type LabelMappingMode = 'auto' | '0t1d' | '0d1t';

const app = document.querySelector<HTMLDivElement>('#app');

if (!app) {
  throw new Error('App root not found.');
}

app.innerHTML = `
  <main class="shell">
    <header class="hero">
      <p class="eyebrow">Transformer Inference Studio</p>
      <h1>Automated Deception Classifier</h1>
      <p class="subtitle">
        Upload your model descriptor, run live predictions on single statements, and batch-score CSV files with metrics and visual analytics.
      </p>
    </header>

    <section class="card model-card">
      <div class="card-head">
        <h2>1. Upload Model</h2>
        <span id="model-status-pill" class="status-pill">Not loaded</span>
      </div>
      <p class="hint">
        Upload your <strong>trained model as a .zip</strong> (must contain ONNX weights — see README for the one-line export command).
        Or upload a <strong>.json descriptor</strong> to stream a Hugging Face remote model instead.
      </p>
      <div class="grid-two">
        <label class="control control-file">
          <span>Model file (.zip with ONNX weights, or .json descriptor)</span>
          <input id="model-config-input" type="file" accept="application/json,.json,application/zip,.zip" />
        </label>
        <label class="control">
          <span>Label Mapping</span>
          <select id="label-mapping-select">
            <option value="auto">Auto — read from config.json in ZIP</option>
            <option value="0t1d" selected>0 = Truthful · 1 = Deceptive &nbsp;(training-pipeline default)</option>
            <option value="0d1t">0 = Deceptive · 1 = Truthful</option>
          </select>
        </label>
      </div>
      <p id="detected-mapping-text" class="hint"></p>
      <div id="upload-progress-wrap" class="upload-progress-wrap hidden">
        <div class="upload-progress-track">
          <div id="upload-progress-bar" class="upload-progress-fill"></div>
        </div>
        <p id="upload-progress-label" class="upload-progress-label">0%</p>
      </div>
      <p id="model-status-text" class="status-text">Waiting for model upload.</p>
    </section>

    <section class="card">
      <div class="card-head">
        <h2>2. Single Statement Prediction</h2>
      </div>
      <label class="control">
        <span>Statement</span>
        <textarea id="statement-input" rows="6" placeholder="Write a statement to evaluate..." maxlength="4000"></textarea>
      </label>
      <div class="meta-row">
        <p id="word-counter">0 words | 0 characters</p>
        <button id="predict-text-btn" class="btn btn-primary" type="button">Predict Statement</button>
      </div>

      <div id="prediction-card" class="prediction hidden" aria-live="polite">
        <div class="prediction-top">
          <strong id="prediction-label">-</strong>
          <span id="prediction-confidence">0.00%</span>
        </div>
        <div class="confidence-track">
          <div id="confidence-bar" class="confidence-fill"></div>
        </div>
      </div>
    </section>

    <section class="card">
      <div class="card-head">
        <h2>3. CSV Batch Prediction + Evaluation</h2>
      </div>
      <p class="hint">
        Need a quick test file? Try
        <a href="/sample-statements.csv" target="_blank" rel="noreferrer">sample-statements.csv</a>.
      </p>
      <div class="grid-two">
        <label class="control control-file">
          <span>CSV File</span>
          <input id="csv-input" type="file" accept=".csv,text/csv" />
        </label>
        <label class="control">
          <span>Text Column</span>
          <select id="csv-text-column" disabled></select>
        </label>
        <label class="control">
          <span>Ground Truth Column (optional)</span>
          <select id="csv-truth-column" disabled></select>
        </label>
        <label class="control">
          <span>Truthful Ground Truth Value</span>
          <input id="truthful-value" type="text" value="truthful" />
        </label>
        <label class="control">
          <span>Deceptive Ground Truth Value</span>
          <input id="deceptive-value" type="text" value="deceptive" />
        </label>
      </div>

      <div class="meta-row">
        <p id="csv-status" class="status-text">No CSV loaded.</p>
        <div class="button-group">
          <button id="predict-csv-btn" class="btn btn-primary" type="button" disabled>Predict CSV</button>
          <button id="download-csv-btn" class="btn" type="button" disabled>Download Labeled CSV</button>
        </div>
      </div>

      <div class="metrics-grid">
        <article class="metric-box"><h3>Accuracy</h3><p id="m-accuracy">-</p></article>
        <article class="metric-box"><h3>F1 (Macro)</h3><p id="m-f1-macro">-</p></article>
        <article class="metric-box"><h3>Precision (Truthful)</h3><p id="m-precision">-</p></article>
        <article class="metric-box"><h3>Recall (Truthful)</h3><p id="m-recall">-</p></article>
      </div>

      <div class="chart-grid">
        <article class="chart-box">
          <h3>Predicted Label Distribution</h3>
          <canvas id="distribution-chart" aria-label="Prediction distribution chart"></canvas>
        </article>
        <article class="chart-box">
          <h3>Average Confidence by Predicted Class</h3>
          <canvas id="confidence-chart" aria-label="Confidence chart"></canvas>
        </article>
      </div>
    </section>
  </main>
`;

const modelConfigInput = document.querySelector<HTMLInputElement>('#model-config-input');
const modelStatusText = document.querySelector<HTMLParagraphElement>('#model-status-text');
const modelStatusPill = document.querySelector<HTMLSpanElement>('#model-status-pill');
const statementInput = document.querySelector<HTMLTextAreaElement>('#statement-input');
const wordCounter = document.querySelector<HTMLParagraphElement>('#word-counter');
const predictTextBtn = document.querySelector<HTMLButtonElement>('#predict-text-btn');
const predictionCard = document.querySelector<HTMLDivElement>('#prediction-card');
const predictionLabel = document.querySelector<HTMLElement>('#prediction-label');
const predictionConfidence = document.querySelector<HTMLElement>('#prediction-confidence');

const confidenceBar = document.querySelector<HTMLDivElement>('#confidence-bar');

const csvInput = document.querySelector<HTMLInputElement>('#csv-input');
const csvStatus = document.querySelector<HTMLParagraphElement>('#csv-status');
const csvTextColumn = document.querySelector<HTMLSelectElement>('#csv-text-column');
const csvTruthColumn = document.querySelector<HTMLSelectElement>('#csv-truth-column');
const truthfulValue = document.querySelector<HTMLInputElement>('#truthful-value');
const deceptiveValue = document.querySelector<HTMLInputElement>('#deceptive-value');
const predictCsvBtn = document.querySelector<HTMLButtonElement>('#predict-csv-btn');
const downloadCsvBtn = document.querySelector<HTMLButtonElement>('#download-csv-btn');

const mAccuracy = document.querySelector<HTMLParagraphElement>('#m-accuracy');
const mF1Macro = document.querySelector<HTMLParagraphElement>('#m-f1-macro');
const mPrecision = document.querySelector<HTMLParagraphElement>('#m-precision');
const mRecall = document.querySelector<HTMLParagraphElement>('#m-recall');

const distributionChartCanvas = document.querySelector<HTMLCanvasElement>('#distribution-chart');
const confidenceChartCanvas = document.querySelector<HTMLCanvasElement>('#confidence-chart');
const labelMappingSelect = document.querySelector<HTMLSelectElement>('#label-mapping-select');
const detectedMappingText = document.querySelector<HTMLParagraphElement>('#detected-mapping-text');
const uploadProgressWrap = document.querySelector<HTMLDivElement>('#upload-progress-wrap');
const uploadProgressBar = document.querySelector<HTMLDivElement>('#upload-progress-bar');
const uploadProgressLabel = document.querySelector<HTMLParagraphElement>('#upload-progress-label');

if (
  !modelConfigInput ||
  !modelStatusText ||
  !modelStatusPill ||
  !statementInput ||
  !wordCounter ||
  !predictTextBtn ||
  !predictionCard ||
  !predictionLabel ||
  !predictionConfidence ||
  !confidenceBar ||
  !csvInput ||
  !csvStatus ||
  !csvTextColumn ||
  !csvTruthColumn ||
  !truthfulValue ||
  !deceptiveValue ||
  !predictCsvBtn ||
  !downloadCsvBtn ||
  !mAccuracy ||
  !mF1Macro ||
  !mPrecision ||
  !mRecall ||
  !distributionChartCanvas ||
  !confidenceChartCanvas ||
  !labelMappingSelect ||
  !detectedMappingText ||
  !uploadProgressWrap ||
  !uploadProgressBar ||
  !uploadProgressLabel
) {
  throw new Error('Some UI elements could not be initialized.');
}

let modelLoaded = false;
let parsedCsvRows: CsvRow[] = [];
let labeledCsvRows: Array<Record<string, string | number>> = [];
let distributionChart: Chart | null = null;
let confidenceChart: Chart | null = null;
let labelMappingMode: LabelMappingMode = '0t1d';
let detectedId2Label: Record<string, string> | null = null;
let uploadStartedAt: number | null = null;
let uploadTimer: ReturnType<typeof setInterval> | null = null;
let uploadLabelBase = '0%';

const formatElapsedTime = (elapsedMs: number): string => {
  const totalSeconds = Math.max(0, Math.floor(elapsedMs / 1000));
  const minutes = Math.floor(totalSeconds / 60)
    .toString()
    .padStart(2, '0');
  const seconds = (totalSeconds % 60).toString().padStart(2, '0');
  return `${minutes}:${seconds}`;
};

const updateUploadProgressLabel = (): void => {
  if (uploadStartedAt === null) {
    uploadProgressLabel.textContent = uploadLabelBase;
    return;
  }

  const elapsed = formatElapsedTime(Date.now() - uploadStartedAt);
  uploadProgressLabel.textContent = `${uploadLabelBase} · elapsed ${elapsed}`;
};

const startUploadTimer = (): void => {
  uploadStartedAt = Date.now();
  if (uploadTimer) {
    clearInterval(uploadTimer);
  }
  uploadTimer = setInterval(updateUploadProgressLabel, 1000);
  updateUploadProgressLabel();
};

const stopUploadTimer = (): string => {
  if (uploadTimer) {
    clearInterval(uploadTimer);
    uploadTimer = null;
  }

  if (uploadStartedAt === null) {
    return '00:00';
  }

  const elapsed = formatElapsedTime(Date.now() - uploadStartedAt);
  uploadStartedAt = null;
  return elapsed;
};

const setUploadProgress = (current: number, total: number, label?: string): void => {
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;
  uploadProgressWrap.classList.remove('hidden');
  uploadProgressBar.classList.remove('indeterminate');
  uploadProgressBar.style.width = `${pct}%`;
  uploadLabelBase = label ?? `${current} / ${total} files cached (${pct}%)`;
  updateUploadProgressLabel();
};

const setUploadProgressIndeterminate = (label: string): void => {
  uploadProgressWrap.classList.remove('hidden');
  uploadProgressBar.classList.add('indeterminate');
  uploadProgressBar.style.width = '100%';
  uploadLabelBase = label;
  updateUploadProgressLabel();
};

const resetUploadProgress = (): void => {
  uploadProgressWrap.classList.add('hidden');
  uploadProgressBar.classList.remove('indeterminate');
  uploadProgressBar.style.width = '0%';
  if (uploadTimer) {
    clearInterval(uploadTimer);
    uploadTimer = null;
  }
  uploadStartedAt = null;
  uploadLabelBase = '0%';
  uploadProgressLabel.textContent = '0%';
};

// ---------------------------------------------------------------------------
// Backend API helpers
// ---------------------------------------------------------------------------

/** Upload a model ZIP to the backend via XHR so we get real byte-progress. */
const uploadZipToBackend = (
  file: File,
  onProgress: (msg: string, loaded: number, total: number) => void,
): Promise<{ id2label: Record<string, string> }> =>
  new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append('file', file);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        onProgress(
          `Uploading to backend: ${Math.round((e.loaded / e.total) * 100)}%`,
          e.loaded,
          e.total,
        );
      }
    };

    xhr.onload = () => {
      if (xhr.status === 200) {
        const body = JSON.parse(xhr.responseText) as { id2label?: Record<string, string> };
        resolve({ id2label: body.id2label ?? {} });
      } else {
        const detail = (() => {
          try {
            return (JSON.parse(xhr.responseText) as { detail?: string }).detail ?? xhr.responseText;
          } catch {
            return xhr.responseText;
          }
        })();
        reject(new Error(detail));
      }
    };

    xhr.onerror = () => reject(new Error('Network error — is the backend running on port 8000?'));
    xhr.ontimeout = () => reject(new Error('Upload timed out.'));

    xhr.open('POST', `${BACKEND_URL}/upload`);
    xhr.send(form);
  });

const setModelStatus = (text: string, mode: 'idle' | 'loading' | 'ready' | 'error'): void => {
  modelStatusText.textContent = text;
  modelStatusPill.textContent = mode === 'ready' ? 'Loaded' : mode === 'loading' ? 'Loading' : mode === 'error' ? 'Error' : 'Not loaded';
  modelStatusPill.className = `status-pill ${mode}`;
};

const setCsvStatus = (text: string): void => {
  csvStatus.textContent = text;
};

const countWords = (value: string): number => {
  const trimmed = value.trim();
  if (!trimmed) {
    return 0;
  }
  return trimmed.split(/\s+/).length;
};

const updateWordCounter = (): void => {
  const text = statementInput.value;
  wordCounter.textContent = `${countWords(text)} words | ${text.length} characters`;
};



// ---------------------------------------------------------------------------
// Label mapping
// ---------------------------------------------------------------------------

const updateDetectedMappingText = (): void => {
  if (!detectedId2Label || labelMappingMode !== 'auto') {
    detectedMappingText.textContent = '';
    return;
  }
  const entries = Object.entries(detectedId2Label)
    .map(([k, v]) => `LABEL_${k} → ${v}`)
    .join(', ');
  detectedMappingText.textContent = `Auto-detected from config.json: ${entries}`;
};

const isTruthyLabel = (rawLabel: string): DeceptionLabel => {
  const value = rawLabel.toLowerCase();

  // Auto mode: resolve via id2label from config.json.
  if (labelMappingMode === 'auto' && detectedId2Label) {
    const numericKey = value.replace(/^label_/, '');
    const mapped = (detectedId2Label[numericKey] ?? '').toLowerCase();
    if (mapped.includes('truth') || mapped.includes('honest')) return 'truthful';
    if (mapped.includes('decept') || mapped.includes('dishonest') || mapped.includes('lie')) return 'deceptive';
  }

  // Explicit mapping: training-pipeline semantics (0 = truthful, 1 = deceptive).
  if (labelMappingMode === '0t1d' || (labelMappingMode === 'auto' && !detectedId2Label)) {
    if (value === 'label_0' || value === '0') return 'truthful';
    if (value === 'label_1' || value === '1') return 'deceptive';
  }

  // Explicit mapping: evaluation-output semantics (0 = deceptive, 1 = truthful).
  if (labelMappingMode === '0d1t') {
    if (value === 'label_0' || value === '0') return 'deceptive';
    if (value === 'label_1' || value === '1') return 'truthful';
  }

  // Keyword fallback.
  if (value.includes('truth') || value.includes('honest')) return 'truthful';
  if (value.includes('decept') || value.includes('dishonest') || value.includes('lie')) return 'deceptive';
  return value.includes('1') ? 'truthful' : 'deceptive';
};

const classifyStatement = async (text: string): Promise<Prediction> => {
  if (!modelLoaded) {
    throw new Error('Model is not loaded.');
  }

  const response = await fetch(`${BACKEND_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const detail = await response.json().then((d: { detail?: string }) => d.detail ?? 'Inference failed').catch(() => 'Inference failed');
    throw new Error(detail);
  }

  const result = await response.json() as { label: string; score: number };

  return {
    label: isTruthyLabel(result.label),
    confidence: result.score,
    rawLabel: result.label,
  };
};

const renderSinglePrediction = (prediction: Prediction): void => {
  predictionCard.classList.remove('hidden');
  predictionLabel.textContent = prediction.label.toUpperCase();
  predictionLabel.className = prediction.label === 'truthful' ? 'label-truthful' : 'label-deceptive';
  predictionConfidence.textContent = `${(prediction.confidence * 100).toFixed(2)}%`;
  confidenceBar.style.width = `${Math.min(100, Math.max(0, prediction.confidence * 100))}%`;
  confidenceBar.className = prediction.label === 'truthful' ? 'confidence-fill truthful' : 'confidence-fill deceptive';
};

const fillSelectWithColumns = (select: HTMLSelectElement, columns: string[], allowNone = false): void => {
  select.innerHTML = '';
  if (allowNone) {
    const noneOption = document.createElement('option');
    noneOption.value = '';
    noneOption.textContent = 'None';
    select.append(noneOption);
  }
  columns.forEach((column) => {
    const option = document.createElement('option');
    option.value = column;
    option.textContent = column;
    select.append(option);
  });
  select.disabled = columns.length === 0;
};

const normalizeGroundTruth = (value: string, truthfulRaw: string, deceptiveRaw: string): DeceptionLabel | null => {
  const current = value.trim().toLowerCase();
  if (!current) {
    return null;
  }
  if (current === truthfulRaw.trim().toLowerCase()) {
    return 'truthful';
  }
  if (current === deceptiveRaw.trim().toLowerCase()) {
    return 'deceptive';
  }
  if (['1', 'true', 'truthful', 'honest'].includes(current)) {
    return 'truthful';
  }
  if (['0', 'false', 'deceptive', 'dishonest'].includes(current)) {
    return 'deceptive';
  }
  return null;
};

const safeDivide = (numerator: number, denominator: number): number => {
  if (denominator === 0) {
    return 0;
  }
  return numerator / denominator;
};

const calculateMetrics = (groundTruth: DeceptionLabel[], predictions: DeceptionLabel[]): Metrics => {
  let truthfulTrue = 0;
  let truthfulFalse = 0;
  let deceptiveTrue = 0;
  let deceptiveFalse = 0;

  groundTruth.forEach((truth, index) => {
    const prediction = predictions[index];
    if (truth === 'truthful' && prediction === 'truthful') truthfulTrue += 1;
    if (truth === 'truthful' && prediction === 'deceptive') truthfulFalse += 1;
    if (truth === 'deceptive' && prediction === 'deceptive') deceptiveTrue += 1;
    if (truth === 'deceptive' && prediction === 'truthful') deceptiveFalse += 1;
  });

  const accuracy = safeDivide(truthfulTrue + deceptiveTrue, groundTruth.length);
  const precisionTruthful = safeDivide(truthfulTrue, truthfulTrue + deceptiveFalse);
  const recallTruthful = safeDivide(truthfulTrue, truthfulTrue + truthfulFalse);
  const f1Truthful = safeDivide(2 * precisionTruthful * recallTruthful, precisionTruthful + recallTruthful);
  const precisionDeceptive = safeDivide(deceptiveTrue, deceptiveTrue + truthfulFalse);
  const recallDeceptive = safeDivide(deceptiveTrue, deceptiveTrue + deceptiveFalse);
  const f1Deceptive = safeDivide(2 * precisionDeceptive * recallDeceptive, precisionDeceptive + recallDeceptive);

  return {
    accuracy,
    precisionTruthful,
    recallTruthful,
    f1Truthful,
    precisionDeceptive,
    recallDeceptive,
    f1Deceptive,
    f1Macro: (f1Truthful + f1Deceptive) / 2,
    support: groundTruth.length,
    confusion: {
      truthfulTrue,
      truthfulFalse,
      deceptiveTrue,
      deceptiveFalse,
    },
  };
};

const renderMetrics = (metrics: Metrics | null): void => {
  if (!metrics) {
    mAccuracy.textContent = 'N/A';
    mF1Macro.textContent = 'N/A';
    mPrecision.textContent = 'N/A';
    mRecall.textContent = 'N/A';
    return;
  }

  mAccuracy.textContent = `${(metrics.accuracy * 100).toFixed(2)}%`;
  mF1Macro.textContent = metrics.f1Macro.toFixed(4);
  mPrecision.textContent = metrics.precisionTruthful.toFixed(4);
  mRecall.textContent = metrics.recallTruthful.toFixed(4);
};

const renderCharts = (predictions: Prediction[]): void => {
  const truthful = predictions.filter((item) => item.label === 'truthful');
  const deceptive = predictions.filter((item) => item.label === 'deceptive');
  const truthfulAvg = safeDivide(
    truthful.reduce((sum, item) => sum + item.confidence, 0),
    truthful.length,
  );
  const deceptiveAvg = safeDivide(
    deceptive.reduce((sum, item) => sum + item.confidence, 0),
    deceptive.length,
  );

  distributionChart?.destroy();
  confidenceChart?.destroy();

  distributionChart = new Chart(distributionChartCanvas, {
    type: 'doughnut',
    data: {
      labels: ['Deceptive', 'Truthful'],
      datasets: [
        {
          data: [deceptive.length, truthful.length],
          backgroundColor: ['#d42f1a', '#1f9d55'],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          labels: {
            color: '#f2f5f7',
          },
        },
      },
    },
  });

  confidenceChart = new Chart(confidenceChartCanvas, {
    type: 'bar',
    data: {
      labels: ['Deceptive', 'Truthful'],
      datasets: [
        {
          label: 'Average Confidence',
          data: [deceptiveAvg * 100, truthfulAvg * 100],
          backgroundColor: ['#ef6448', '#32be73'],
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: {
            callback: (value) => `${value}%`,
            color: '#c9d4dd',
          },
          grid: {
            color: 'rgba(201, 212, 221, 0.2)',
          },
        },
        x: {
          ticks: {
            color: '#c9d4dd',
          },
          grid: {
            display: false,
          },
        },
      },
      plugins: {
        legend: {
          labels: {
            color: '#f2f5f7',
          },
        },
      },
    },
  });
};

const downloadLabeledCsv = (): void => {
  if (!labeledCsvRows.length) {
    return;
  }

  const csv = Papa.unparse(labeledCsvRows);
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);

  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `labeled_predictions_${Date.now()}.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
};

// ---------------------------------------------------------------------------
// Model upload handler — two paths: local ZIP or remote JSON descriptor
// ---------------------------------------------------------------------------

labelMappingSelect.addEventListener('change', () => {
  labelMappingMode = labelMappingSelect.value as LabelMappingMode;
  updateDetectedMappingText();
});

modelConfigInput.addEventListener('change', async () => {
  const file = modelConfigInput.files?.[0];
  if (!file) return;

  const lowerName = file.name.toLowerCase();

  try {
    modelLoaded = false;
    detectedId2Label = null;
    resetUploadProgress();
    startUploadTimer();
    setModelStatus('Connecting to backend…', 'loading');

    if (lowerName.endsWith('.zip')) {
      // ── Local model path — upload ZIP to backend ─────────────────────────
      setModelStatus('Uploading model to backend…', 'loading');
      setUploadProgress(0, 1, 'Starting upload…');

      const result = await uploadZipToBackend(file, (msg, loaded, total) => {
        setModelStatus(msg, 'loading');
        setUploadProgress(loaded, total);
      });

      // Upload done; backend is now loading the model weights.
      setUploadProgressIndeterminate('Backend loading model weights…');
      setModelStatus('Backend initialising model…', 'loading');

      if (Object.keys(result.id2label).length) {
        detectedId2Label = result.id2label;
        updateDetectedMappingText();
      }

      modelLoaded = true;

      const elapsed = stopUploadTimer();
      setUploadProgress(1, 1, `Ready — loaded in ${elapsed}`);
      setModelStatus(`Model ready · ${file.name}`, 'ready');

    } else if (lowerName.endsWith('.json')) {
      // ── Remote HF descriptor path — let backend download from HF ─────────
      const descriptor = JSON.parse(await file.text()) as { modelId?: string };
      if (!descriptor.modelId) {
        throw new Error('Descriptor must include "modelId".');
      }

      setModelStatus(`Asking backend to load: ${descriptor.modelId}…`, 'loading');
      setUploadProgressIndeterminate(`Downloading ${descriptor.modelId} from HuggingFace…`);
      uploadProgressWrap.classList.remove('hidden');

      const response = await fetch(`${BACKEND_URL}/load_remote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modelId: descriptor.modelId }),
      });

      if (!response.ok) {
        const detail = await response.json().then((d: { detail?: string }) => d.detail ?? 'Failed').catch(() => 'Failed');
        throw new Error(detail);
      }

      modelLoaded = true;
      stopUploadTimer();
      resetUploadProgress();
      setModelStatus(`Remote model ready: ${descriptor.modelId}`, 'ready');

    } else {
      throw new Error('Unsupported file. Upload a .zip (trained model) or .json (HF descriptor).');
    }

  } catch (error) {
    stopUploadTimer();
    const message = error instanceof Error ? error.message : 'Unknown error';
    setModelStatus(`Failed to load model: ${message}`, 'error');
    modelLoaded = false;
  }
});

statementInput.addEventListener('input', updateWordCounter);

predictTextBtn.addEventListener('click', async () => {
  const statement = statementInput.value.trim();
  if (!statement) {
    setModelStatus('Please write a statement first.', 'error');
    return;
  }

  if (!modelLoaded) {
    setModelStatus('Load a model before predicting.', 'error');
    return;
  }

  try {
    const prediction = await classifyStatement(statement);
    renderSinglePrediction(prediction);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Prediction failed.';
    setModelStatus(message, 'error');
  }
});

csvInput.addEventListener('change', () => {
  const file = csvInput.files?.[0];
  if (!file) {
    return;
  }

  Papa.parse<CsvRow>(file, {
    header: true,
    skipEmptyLines: true,
    complete: (results) => {
      if (results.errors.length > 0) {
        setCsvStatus(`CSV parse error: ${results.errors[0]?.message ?? 'Unknown parse error'}`);
        parsedCsvRows = [];
        return;
      }

      parsedCsvRows = results.data;
      const firstRow = parsedCsvRows[0];
      const columns = firstRow ? Object.keys(firstRow) : [];

      fillSelectWithColumns(csvTextColumn, columns);
      fillSelectWithColumns(csvTruthColumn, columns, true);
      predictCsvBtn.disabled = !columns.length;
      setCsvStatus(`Loaded ${parsedCsvRows.length} rows and ${columns.length} columns.`);
    },
    error: (error) => {
      setCsvStatus(`CSV parse failed: ${error.message}`);
    },
  });
});

predictCsvBtn.addEventListener('click', async () => {
  if (!modelLoaded) {
    setModelStatus('Load a model before running batch prediction.', 'error');
    return;
  }

  if (!parsedCsvRows.length) {
    setCsvStatus('Please load a CSV first.');
    return;
  }

  const textColumn = csvTextColumn.value;
  const truthColumn = csvTruthColumn.value;
  if (!textColumn) {
    setCsvStatus('Select a text column for inference.');
    return;
  }

  try {
    predictCsvBtn.disabled = true;
    downloadCsvBtn.disabled = true;

    const predictions: Prediction[] = [];
    const groundTruth: DeceptionLabel[] = [];
    const predictedLabels: DeceptionLabel[] = [];

    labeledCsvRows = [];

    for (let index = 0; index < parsedCsvRows.length; index += 1) {
      const row = parsedCsvRows[index];
      const text = String(row[textColumn] ?? '').trim();

      let prediction: Prediction;
      if (text) {
        prediction = await classifyStatement(text);
      } else {
        prediction = {
          label: 'deceptive',
          confidence: 0,
          rawLabel: 'EMPTY_INPUT',
        };
      }

      predictions.push(prediction);
      predictedLabels.push(prediction.label);

      const enrichedRow: Record<string, string | number> = {
        ...row,
        label_numeric: prediction.label === 'truthful' ? 1 : 0,
        label_string: prediction.label,
        confidence: Number(prediction.confidence.toFixed(6)),
      };

      if (truthColumn) {
        const normalizedTruth = normalizeGroundTruth(
          String(row[truthColumn] ?? ''),
          truthfulValue.value,
          deceptiveValue.value,
        );
        if (normalizedTruth) {
          groundTruth.push(normalizedTruth);
        }
      }

      labeledCsvRows.push(enrichedRow);

      if ((index + 1) % 20 === 0 || index + 1 === parsedCsvRows.length) {
        setCsvStatus(`Predicting rows: ${index + 1}/${parsedCsvRows.length}`);
      }
    }

    const metrics = truthColumn && groundTruth.length === predictedLabels.length
      ? calculateMetrics(groundTruth, predictedLabels)
      : null;

    renderMetrics(metrics);
    renderCharts(predictions);
    downloadCsvBtn.disabled = false;

    if (metrics) {
      setCsvStatus(
        `Completed ${parsedCsvRows.length} rows. Accuracy ${(metrics.accuracy * 100).toFixed(2)}%, Macro F1 ${metrics.f1Macro.toFixed(4)}.`,
      );
    } else if (truthColumn) {
      setCsvStatus(
        `Completed ${parsedCsvRows.length} rows. Metrics skipped because not all rows had mappable ground truth labels.`,
      );
    } else {
      setCsvStatus(`Completed ${parsedCsvRows.length} rows. Metrics skipped (no ground truth column selected).`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Batch prediction failed.';
    setCsvStatus(message);
  } finally {
    predictCsvBtn.disabled = false;
  }
});

downloadCsvBtn.addEventListener('click', downloadLabeledCsv);
updateWordCounter();
