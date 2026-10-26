from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import dataset, health, model, predict
from app.core.errors import register_error_handlers
from app.core.logging import setup_logging
from app.core.state import state

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load()
    yield


app = FastAPI(
    title="Churn prediction service",
    description="Обучение модели оттока клиентов и предсказание вероятности ухода.",
    version="1.0.0",
    lifespan=lifespan,
)
register_error_handlers(app)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(dataset.router)
app.include_router(model.router)
