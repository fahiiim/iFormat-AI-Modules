"""Aggregate all version-one API routers."""

from fastapi import APIRouter

from app.api.v1.endpoints.ai import router as ai_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(ai_router)
