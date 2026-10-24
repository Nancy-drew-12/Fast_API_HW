import json
from pathlib import Path

HISTORY_PATH = Path(__file__).parent / "models" / "training_history.json"


def load_history(path: Path = HISTORY_PATH) -> list[dict]:
    """Читает историю обучений из JSON. Нет файла или он повреждён — пустая история."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def add_training_record(record: dict, path: Path = HISTORY_PATH) -> None:
    """Добавляет запись об обучении в конец истории и сохраняет файл."""
    history = load_history(path)
    history.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def filter_history(history: list[dict], model_type: str | None = None) -> list[dict]:
    """Оставляет записи только нужного типа модели (если тип задан)."""
    if model_type is None:
        return history
    return [record for record in history if record["model_type"] == model_type]


def best_record(history: list[dict], metric: str = "f1") -> dict | None:
    """Запись с лучшим значением метрики — чтобы сравнивать настройки."""
    if not history:
        return None
    return max(history, key=lambda record: record["metrics"].get(metric, 0))