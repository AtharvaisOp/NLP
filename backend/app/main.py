"""FastAPI application entry point for MahaPulse."""

from fastapi import FastAPI

from .api.routes import router


app = FastAPI(
    title="MahaPulse NLP API",
    description="API boundary for Marathi and Marathi-English NLP analysis.",
    version="0.1.0",
)
app.include_router(router)
