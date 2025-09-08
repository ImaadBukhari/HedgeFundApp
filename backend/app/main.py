from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import users

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router, prefix="/api/users")

# Add admin routes for testing
@app.get("/admin/test-quarterly")
async def test_quarterly_system():
    """Test endpoint to generate and send quarterly emails"""
    import requests
    
    # Generate recommendations
    gen_response = requests.post("http://localhost:8000/api/users/admin/generate-rebalance-recommendations")
    
    # Send emails
    email_response = requests.post("http://localhost:8000/api/users/admin/send-rebalance-emails")
    
    return {
        "generation": gen_response.json() if gen_response.status_code == 200 else "Failed",
        "emails": email_response.json() if email_response.status_code == 200 else "Failed"
    }