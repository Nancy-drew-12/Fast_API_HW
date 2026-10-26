from fastapi import APIRouter, Query

from app.api.deps import require_dataset
from app.ml.dataset import get_info, to_rows
from app.ml.preprocessing import get_split_info
from app.schemas import DatasetRowChurn

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/preview")
def dataset_preview(n: int = Query(5, ge=1, le=100)) -> list[DatasetRowChurn]:
    return to_rows(require_dataset().head(n))


@router.get("/info")
def dataset_info():
    return get_info(require_dataset())


@router.get("/split-info")
def dataset_split_info():
    return get_split_info(require_dataset())
