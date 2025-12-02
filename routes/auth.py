from lib import supabase
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional, Any, cast

router = APIRouter(prefix="/api/auth", tags=["auth"])


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


class SignUpRequest(BaseModel):
    email: EmailStr
    password: str
    firstName: str
    lastName: str


class AuthResponse(BaseModel):
    user: Optional[dict]
    session: Optional[dict]
    error: Optional[str] = None


class UserProfileResponse(BaseModel):
    id: str
    auth_id: str
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    has_completed_signup: bool
    created_at: str
    updated_at: str


@router.post("/signin", response_model=AuthResponse)
async def sign_in(credentials: SignInRequest):
    """Sign in an existing user with email and password"""
    try:
        response = supabase.auth.sign_in_with_password(
            {"email": credentials.email, "password": credentials.password}
        )

        if response.user:
            session_data = None
            if response.session is not None:
                session_data = (
                    response.session.model_dump()
                    if hasattr(response.session, "model_dump")
                    else response.session.__dict__
                )
            return AuthResponse(
                user=response.user.model_dump()
                if hasattr(response.user, "model_dump")
                else response.user.__dict__,
                session=session_data,
                error=None,
            )
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")

    except Exception as e:
        error_message = str(e)
        if "Invalid login credentials" in error_message:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        raise HTTPException(
            status_code=500, detail=f"Authentication error: {error_message}"
        )


@router.post("/signup", response_model=AuthResponse)
async def sign_up(data: SignUpRequest):
    """Sign up a new user with email and password"""
    try:
        # Create auth user
        auth_response = supabase.auth.sign_up(
            {"email": data.email, "password": data.password}
        )

        if not auth_response.user:
            raise HTTPException(status_code=400, detail="Failed to create user")

        # Update user profile with first and last name
        supabase.table("users").update(
            {"first_name": data.firstName, "last_name": data.lastName}
        ).eq("auth_id", auth_response.user.id).execute()

        return AuthResponse(
            user=auth_response.user.model_dump()
            if hasattr(auth_response.user, "model_dump")
            else auth_response.user.__dict__,
            session=None,
            error=None,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signup error: {str(e)}")


@router.post("/signout")
async def sign_out():
    """Sign out the current user"""
    try:
        supabase.auth.sign_out()
        return {"message": "Successfully signed out"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sign out error: {str(e)}")


@router.get("/profile/{auth_id}", response_model=UserProfileResponse)
async def get_user_profile(auth_id: str):
    """Get user profile by auth ID"""
    try:
        response = (
            supabase.table("users")
            .select("*")
            .eq("auth_id", auth_id)
            .single()
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=404, detail="User profile not found")

        response_data = cast(dict[str, Any], response.data)
        return UserProfileResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        if "Cannot coerce" in str(e):
            raise HTTPException(status_code=404, detail="User profile not found")
        else:
            raise HTTPException(
                status_code=500, detail=f"Error fetching profile: {str(e)}"
            )


@router.patch("/profile/{auth_id}/complete-signup")
async def complete_signup(auth_id: str):
    """Mark user signup as completed"""
    try:
        supabase.table("users").update({"has_completed_signup": True}).eq(
            "auth_id", auth_id
        ).execute()

        return {"message": "Signup completed successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error completing signup: {str(e)}"
        )
