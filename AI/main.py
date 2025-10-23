from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routers import general, generation

app = FastAPI()

app.mount("/generated_images", StaticFiles(directory="app/generated_images"), name="generated_images")

app.include_router(general.router)
app.include_router(generation.router)