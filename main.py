from fastapi import FastAPI
from app.api.routers import general, generation

app = FastAPI()

app.include_router(general.router)
app.include_router(generation.router)