from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.db.database import supabase
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List
import json

router = APIRouter()

class UserSignup(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class AllocationUpdate(BaseModel):
    allocations: dict

# Basic user endpoints
@router.post("/signup")
def signup(user: UserSignup):
    try:
        # Check if user already exists
        existing = supabase.table("users").select("id").eq("email", user.email).execute()
        if existing.data:
            raise HTTPException(status_code=400, detail="User already exists")

        # Create new user with default allocations
        default_allocations = {
            "ijr": 3000,
            "leveraged_midcap": 3000,
            "trending_value": 4000
        }

        result = supabase.table("users").insert({
            "email": user.email,
            "password": user.password,  # In production, hash this!
            "allocations": default_allocations
        }).execute()

        if result.data:
            return {"message": "User created successfully", "user": result.data[0]}
        else:
            raise HTTPException(status_code=500, detail="Failed to create user")
            
    except Exception as e:
        if "User already exists" in str(e):
            raise e
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")

@router.post("/login")
def login(user: UserLogin):
    try:
        # Find user by email
        result = supabase.table("users").select("*").eq("email", user.email).execute()
        
        if not result.data:
            raise HTTPException(status_code=401, detail="User not found")
        
        user_data = result.data[0]
        
        # Check password (in production, use proper hashing!)
        if user_data["password"] != user.password:
            raise HTTPException(status_code=401, detail="Invalid password")
        
        # Return user data (excluding password)
        user_response = {
            "id": user_data["id"],
            "email": user_data["email"],
            "allocations": user_data.get("allocations", {})
        }
        
        return {"message": "Login successful", "user": user_response}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@router.get("/{user_id}/allocations")
def get_allocations(user_id: str):
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
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get allocations: {str(e)}")

@router.put("/{user_id}/allocations")
def update_allocations(user_id: str, allocation_data: AllocationUpdate):
    try:
        result = supabase.table("users").update({
            "allocations": allocation_data.allocations
        }).eq("id", user_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"message": "Allocations updated successfully", "allocations": allocation_data.allocations}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update allocations: {str(e)}")