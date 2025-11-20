import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from src.demark_world.server.db import init_db
from src.demark_world.server.worker import worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")

    await init_db()
    logger.info("Database initialized")

    await worker.initialize()

    _ = asyncio.create_task(worker.run())

    logger.info("Application started successfully")

    yield

    logger.info("Shutting down...")
    logger.info("Application shutdown complete")
