"""
Deception Classifier — FastAPI backend.

Endpoints
---------
GET  /health          → liveness check
POST /upload          → accepts a .zip of model weights, loads into memory
POST /load_remote     → accepts { "modelId": "org/repo" }, loads from HuggingFace
POST /predict         → { "text": "..." } → { label, score, all_scores }
"""

import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Deception Classifier Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_pipeline: Any = None          # HuggingFace pipeline instance
_model_tmp_dir: Optional[Path] = None   # Temp dir for current ZIP extract
_model_info: dict = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_model_root(base: Path) -> Path:
    """Walk extracted ZIP until we find a directory that contains config.json."""
    for dirpath, _dirs, filenames in os.walk(base):
        if "config.json" in filenames:
            return Path(dirpath)
    return base


def _load_pipeline_from_dir(model_root: Path) -> Any:
    """Load a HuggingFace pipeline from a local directory.

    Prefers ONNX weights (via optimum) if present, otherwise falls back to
    safetensors/pytorch weights via standard transformers.
    """
    import json
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    # Some models (e.g. ModernBERT) ship with tokenizer_class=TokenizersBackend,
    # which AutoTokenizer cannot resolve.  Patch it in-place before loading.
    tok_cfg_path = model_root / "tokenizer_config.json"
    _patched = False
    if tok_cfg_path.exists():
        with open(tok_cfg_path, encoding="utf-8") as fh:
            tok_cfg = json.load(fh)
        unknown_classes = {"TokenizersBackend"}
        if tok_cfg.get("tokenizer_class") in unknown_classes:
            tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            with open(tok_cfg_path, "w", encoding="utf-8") as fh:
                json.dump(tok_cfg, fh, indent=2, ensure_ascii=False)
            _patched = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_root))
    except Exception:
        # Last-resort fallback: load tokenizer.json directly.
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(model_root / "tokenizer.json"))
    onnx_files = sorted(model_root.glob("*.onnx"))
    safetensors_files = sorted(model_root.glob("*.safetensors"))
    pytorch_files = sorted(model_root.glob("pytorch_model*.bin"))

    if onnx_files:
        # Prefer model.onnx, otherwise take the first .onnx found.
        onnx_file = next((f for f in onnx_files if f.name == "model.onnx"), onnx_files[0])
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import pipeline as hf_pipeline

        ort_model = ORTModelForSequenceClassification.from_pretrained(
            str(model_root), file_name=onnx_file.name
        )
        return hf_pipeline(
            "text-classification",
            model=ort_model,
            tokenizer=tokenizer,
            top_k=None,
            device="cpu",
        )

    if safetensors_files or pytorch_files:
        from transformers import AutoModelForSequenceClassification
        from transformers import pipeline as hf_pipeline

        pt_model = AutoModelForSequenceClassification.from_pretrained(str(model_root))
        return hf_pipeline(
            "text-classification",
            model=pt_model,
            tokenizer=tokenizer,
            top_k=None,
            device="cpu",
        )

    raise ValueError(
        "No ONNX (.onnx) or PyTorch (.safetensors / pytorch_model.bin) weights found in ZIP."
    )


def _read_id2label(model_root: Path) -> dict:
    config_path = model_root / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
            return cfg.get("id2label", {})
    return {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.post("/upload")
async def upload_model(file: UploadFile = File(...)) -> dict:
    """Accept a ZIP of model weights, extract, and load into memory.

    Returns immediately once the model is ready for inference.
    """
    global _pipeline, _model_tmp_dir, _model_info

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted.")

    # Tear down any previously loaded model.
    _pipeline = None
    if _model_tmp_dir and _model_tmp_dir.exists():
        shutil.rmtree(_model_tmp_dir, ignore_errors=True)

    # Extract ZIP.
    tmp = Path(tempfile.mkdtemp(prefix="deception_model_"))
    _model_tmp_dir = tmp

    content = await file.read()
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zf.extractall(tmp)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Invalid ZIP file: {exc}") from exc

    model_root = _find_model_root(tmp)
    id2label = _read_id2label(model_root)

    try:
        _pipeline = _load_pipeline_from_dir(model_root)
    except Exception as exc:  # noqa: BLE001
        _pipeline = None
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc

    _model_info = {"filename": file.filename, "id2label": id2label}
    return {"status": "loaded", "filename": file.filename, "id2label": id2label}


class RemoteModelRequest(BaseModel):
    modelId: str


@app.post("/load_remote")
def load_remote_model(req: RemoteModelRequest) -> dict:
    """Load a model directly from HuggingFace Hub by model ID."""
    global _pipeline, _model_tmp_dir, _model_info

    _pipeline = None
    if _model_tmp_dir and _model_tmp_dir.exists():
        shutil.rmtree(_model_tmp_dir, ignore_errors=True)
        _model_tmp_dir = None

    try:
        from transformers import pipeline as hf_pipeline

        _pipeline = hf_pipeline(
            "text-classification",
            model=req.modelId,
            top_k=None,
            device="cpu",
        )
    except Exception as exc:  # noqa: BLE001
        _pipeline = None
        raise HTTPException(
            status_code=500, detail=f"Failed to load remote model: {exc}"
        ) from exc

    _model_info = {"modelId": req.modelId, "id2label": {}}
    return {"status": "loaded", "modelId": req.modelId}


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if _pipeline is None:
        raise HTTPException(status_code=400, detail="No model loaded. Upload a model first.")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    try:
        results = _pipeline(req.text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    # top_k=None returns a list-of-lists; unwrap if needed.
    scores: list = results[0] if (results and isinstance(results[0], list)) else results
    best: dict = max(scores, key=lambda x: x["score"])

    return {
        "label": best["label"],
        "score": best["score"],
        "all_scores": scores,
    }


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
