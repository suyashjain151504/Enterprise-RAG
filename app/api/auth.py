import psycopg2
from fastapi import APIRouter, HTTPException, Request, status

from app.config import settings
from app.middleware.auth import create_access_token, hash_password, verify_password
from app.middleware.rate_limiter import is_allowed_ip


