from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import users, strategies

app = FastAPI()

# 🔐 Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend dev URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allows Content-Type, Authorization, etc.
)

# 🚀 Register routes
app.include_router(users.router, prefix="/api/users")
app.include_router(strategies.router, prefix="/api/strategies")
