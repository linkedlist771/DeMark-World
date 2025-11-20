from fastapi import FastAPI

from src.demark_world.server.lifespan import lifespan
from src.demark_world.server.router import router


def init_app():
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app
