from fastapi import FastAPI

from app.api import admin, auth

app = FastAPI(title="Adv RAG", version="0.1.0-lesson-0")
app.include_router(admin.router)
app.include_router(auth.router)