from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.db.database import supabase

router = APIRouter()

class UserSignup(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class AllocationUpdate(BaseModel):
    allocations: dict

@router.post("/signup")
def signup(user: UserSignup):
    existing = supabase.table("users").select("id").eq("email", user.email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="User already exists")

    # Default allocations for new users
    default_allocations = {
        "ijr": 3000,
        "leveraged_midcap": 3000,
        "trending_value": 4000
    }

    result = supabase.table("users").insert({
        "email": user.email,
        "password": user.password,
        "allocations": default_allocations
    }).execute()

    return {"message": "User created", "user": result.data[0]}

@router.post("/login")
def login(user: UserLogin):
    match = supabase.table("users").select("*").eq("email", user.email).execute()
    if not match.data:
        raise HTTPException(status_code=401, detail="User not found")

    if match.data[0]["password"] != user.password:
        raise HTTPException(status_code=401, detail="Incorrect password")

    return {"user": match.data[0]}

# NEW ENDPOINTS FOR DASHBOARD
@router.get("/{user_id}/allocations")
def get_allocations(user_id: str):
    """Get user's portfolio allocations"""
    try:
        result = supabase.table("users").select("allocations").eq("id", user_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="User not found")
        
        allocations = result.data[0].get("allocations", {
            "ijr": 3000,
            "leveraged_midcap": 3000,
            "trending_value": 4000
        })
        
        return {"allocations": allocations}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{user_id}/allocations")
def update_allocations(user_id: str, allocation_data: AllocationUpdate):
    """Update user's portfolio allocations"""
    try:
        result = supabase.table("users").update({
            "allocations": allocation_data.allocations
        }).eq("id", user_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"message": "Allocations updated successfully", "allocations": allocation_data.allocations}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))