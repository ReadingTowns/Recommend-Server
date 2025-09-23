from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.db import models, database

@asynccontextmanager
async def lifespan(app: FastAPI):

    models.Base.metadata.create_all(bind=database.engine)
    yield

app = FastAPI(title="Book Recommend API", lifespan=lifespan)
app.include_router(router)