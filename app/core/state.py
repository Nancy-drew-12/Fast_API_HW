"""Состояние сервиса в памяти: загруженный датасет и обученная модель."""
from datetime import datetime

import pandas as pd

from app.core.logging import logger
from app.ml.dataset import load_dataset
from app.ml.model import load_churn_model


class ServiceState:
    def __init__(self) -> None:
        self.dataset: pd.DataFrame | None = None
        self.model_bundle: dict | None = None
        self.started_at = datetime.now().isoformat(timespec="seconds")

    def load(self) -> None:
        """Читает датасет и сохранённую модель при старте сервиса."""
        try:
            self.dataset = load_dataset()
            logger.info("Dataset loaded: %d rows, %d columns", *self.dataset.shape)
        except FileNotFoundError:
            self.dataset = None
            logger.warning("Dataset file not found, service starts without data")

        try:
            self.model_bundle = load_churn_model()
        except Exception:
            self.model_bundle = None
            logger.exception("Failed to load saved model, starting without it")
            return

        if self.model_bundle is None:
            logger.info("No saved model found, train it via POST /model/train")
        else:
            logger.info("Model loaded: %s, trained at %s",
                        self.model_bundle["model_type"], self.model_bundle["trained_at"])


state = ServiceState()
