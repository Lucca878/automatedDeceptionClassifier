# Automated Deception Classifier

A TypeScript web app for deception detection backed by a Python FastAPI inference server.

## Features

- Single statement prediction with live word counter and confidence bar
- CSV batch prediction with automatic enrichment (`label_numeric`, `label_string`, `confidence`)
- Evaluation metrics (accuracy, precision, recall, F1) when ground truth is provided
- Visual analytics with charts

## Architecture

```
Browser (Vite + TypeScript)
        ↕  HTTP / JSON
FastAPI backend (Python)
        ↕  transformers / optimum
Your trained model (safetensors or ONNX)
```

Inference runs entirely on the Python backend — the browser only sends text and receives a label + score. No model weights are ever loaded in the browser.

## Tech Stack

- **Frontend**: Vite + TypeScript, `papaparse`, `chart.js`
- **Backend**: FastAPI, `transformers`, `optimum[onnxruntime]`

## Quick Start

**1. Start the backend**

```bash
pip install fastapi "uvicorn[standard]" python-multipart transformers optimum[onnxruntime]
cd backend
python -m uvicorn app:app --reload --port 8000
```

**2. Start the frontend**

```bash
npm install
npm run dev
```

Open http://localhost:5173.

## Build for Production

```bash
npm run build   # outputs to dist/
npm run preview
```

## How Inference Works

### Path A — Upload your own trained model (ZIP)

Zip your model folder (safetensors **or** ONNX — both work) and upload it in the app.

```bash
# Option 1: original PyTorch/safetensors weights (simplest)
zip -r my_model.zip models/modernbert_trained/model/

# Option 2: ONNX export (faster inference, slightly smaller)
pip install optimum[exporters]
optimum-cli export onnx --model ./models/modernbert_trained/model ./models/modernbert_trained/model_onnx
zip -r my_model_onnx.zip models/modernbert_trained/model_onnx/
```

The ZIP must contain `config.json` and either `model.safetensors` or a `*.onnx` file. The backend auto-detects which format is present.

> **ONNX vs safetensors**: You do not need to convert to ONNX. The backend loads safetensors natively via `transformers`. ONNX gives faster inference via ONNX Runtime but is otherwise equivalent. Use whatever you already have.

### Path B — Load a model from Hugging Face

Upload a `.json` descriptor:

```json
{ "modelId": "owner/repo-name" }
```

The backend downloads and caches the model from the Hub. A sample is at `public/sample-model-config.json`.

### Label Mapping

The app reads `id2label` from `config.json` automatically. If predictions look inverted, change the **Label Mapping** dropdown:

| Option | When to use |
|---|---|
| **Auto — read from config.json** | Default. Uses `id2label` detected from your model. |
| **0 = Truthful · 1 = Deceptive** | If your training encoded `truthful=0, deceptive=1`. |
| **0 = Deceptive · 1 = Truthful** | If your training encoded `deceptive=0, truthful=1`. |

## CSV Batch Prediction

1. Upload a CSV file.
2. Select the text column.
3. (Optional) Select a ground truth column and set the truthful/deceptive raw values.
4. Run prediction.
5. Download the enriched CSV.

Sample input: `public/sample-statements.csv`

Output columns added: `label_numeric`, `label_string`, `confidence`

## Publish to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

## Notes

- Ground truth mapping must be consistent across all rows for full metric calculation.
- If predictions look inverted, change the Label Mapping dropdown.
- The backend keeps the last loaded model in memory; uploading a new ZIP replaces it.
