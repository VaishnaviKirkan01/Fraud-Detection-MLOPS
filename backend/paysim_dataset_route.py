# FraudGuard PaySim dataset endpoint
# Add this code to your existing FastAPI main.py AFTER `app = FastAPI(...)`.
# Keep your existing prediction/database routes unchanged.
import os
from pathlib import Path
import pandas as pd
from fastapi import Query

_DATASET_CACHE = None
_DATASET_COLUMNS = [
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "isFraud"
]

def _load_paysim_preview():
    """Load a small, balanced preview without loading the entire 6.3M-row CSV into RAM."""
    global _DATASET_CACHE
    if _DATASET_CACHE is not None:
        return _DATASET_CACHE

    configured = os.getenv("PAYSIM_DATA_PATH")
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        Path(configured).expanduser() if configured else None,
        project_root / "data" / "raw" / "paysim.csv",
        Path(__file__).resolve().parent / "data" / "raw" / "paysim.csv",
    ]
    csv_path = next((p for p in candidates if p is not None and p.is_file()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "PaySim CSV not found. Set PAYSIM_DATA_PATH to the full path of data/raw/paysim.csv."
        )

    legitimate, fraud = [], []
    # Read in chunks and keep a representative mix of both source labels.
    for chunk in pd.read_csv(csv_path, usecols=_DATASET_COLUMNS, chunksize=100_000):
        for row in chunk.to_dict(orient="records"):
            row["isFraud"] = int(row.get("isFraud", 0))
            target = fraud if row["isFraud"] == 1 else legitimate
            if len(target) < 50:
                target.append(row)
        if len(legitimate) >= 50 and len(fraud) >= 50:
            break

    # If the dataset preview is extremely small or has no fraud rows in scanned chunks,
    # return whatever representative rows were found without inventing labels.
    rows = []
    rows.extend({"dataset_id": f"PS-L-{i+1}", **r} for i, r in enumerate(legitimate))
    rows.extend({"dataset_id": f"PS-F-{i+1}", **r} for i, r in enumerate(fraud))
    _DATASET_CACHE = rows
    return _DATASET_CACHE

@app.get("/api/dataset/transactions")
def dataset_transactions(limit: int = Query(default=100, ge=1, le=100)):
    """Read-only source dataset preview; these are dataset labels, not model predictions."""
    try:
        rows = _load_paysim_preview()
        return {"items": rows[:limit], "count": len(rows), "source": "PaySim dataset"}
    except FileNotFoundError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Could not read PaySim dataset: {exc}") from exc
