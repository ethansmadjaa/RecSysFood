from lib.supabase import supabase
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

router = APIRouter(prefix="/api/preferences", tags=["preferences"])

class MealTypeEnum(str, Enum):
    breakfast_brunch = "breakfast_brunch"
    main_course = "main_course"
    starter_side = "starter_side"
    dessert = "dessert"
    snack = "snack"

class CalorieGoalEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class ProteinGoalEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class UserPreferencesRequest(BaseModel):
    user_id: str
    meal_types: List[MealTypeEnum]
    max_total_time: Optional[int] = None
    calorie_goal: CalorieGoalEnum = CalorieGoalEnum.medium
    protein_goal: ProteinGoalEnum = ProteinGoalEnum.medium
    dietary_restrictions: List[str] = []
    allergy_nuts: bool = False
    allergy_dairy: bool = False
    allergy_egg: bool = False
    allergy_fish: bool = False
    allergy_soy: bool = False

class UserPreferencesResponse(BaseModel):
    user_preferences_id: int
    user_id: str
    meal_types: List[str]
    max_total_time: Optional[int]
    calorie_goal: str
    protein_goal: str
    dietary_restrictions: List[str]
    allergy_nuts: bool
    allergy_dairy: bool
    allergy_egg: bool
    allergy_fish: bool
    allergy_soy: bool
    created_at: str
    updated_at: str

@router.post("/", response_model=UserPreferencesResponse)
async def create_user_preferences(preferences: UserPreferencesRequest):
    """Create or update user preferences"""
    try:
        # Convert meal_types list to PostgreSQL array format
        meal_types_array = [mt.value for mt in preferences.meal_types]

        # Prepare the data
        preferences_data = {
            "user_id": preferences.user_id,
            "meal_types": meal_types_array,
            "max_total_time": preferences.max_total_time,
            "calorie_goal": preferences.calorie_goal.value,
            "protein_goal": preferences.protein_goal.value,
            "dietary_restrictions": preferences.dietary_restrictions,
            "allergy_nuts": preferences.allergy_nuts,
            "allergy_dairy": preferences.allergy_dairy,
            "allergy_egg": preferences.allergy_egg,
            "allergy_fish": preferences.allergy_fish,
            "allergy_soy": preferences.allergy_soy
        }

        # Check if preferences already exist for this user
        existing = supabase.table('user_preferences').select('*').eq('user_id', preferences.user_id).execute()

        if existing.data:
            # Update existing preferences
            response = supabase.table('user_preferences').update(
                preferences_data
            ).eq('user_id', preferences.user_id).execute()
        else:
            # Insert new preferences
            response = supabase.table('user_preferences').insert(
                preferences_data
            ).execute()

        if not response.data:
            raise HTTPException(status_code=400, detail="Failed to save preferences")

        return UserPreferencesResponse(**response.data[0])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving preferences: {str(e)}")

@router.get("/{user_id}", response_model=UserPreferencesResponse)
async def get_user_preferences(user_id: str):
    """Get user preferences by user ID"""
    try:
        response = supabase.table('user_preferences').select('*').eq('user_id', user_id).single().execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="User preferences not found")

        return UserPreferencesResponse(**response.data)

    except HTTPException:
        raise
    except Exception as e:
        if "Cannot coerce" in str(e):
            raise HTTPException(status_code=404, detail="User preferences not found")
        else:
            raise HTTPException(status_code=500, detail=f"Error fetching preferences: {str(e)}")

@router.delete("/{user_id}")
async def delete_user_preferences(user_id: str):
    """Delete user preferences"""
    try:
        supabase.table('user_preferences').delete().eq('user_id', user_id).execute()
        return {"message": "Preferences deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting preferences: {str(e)}")
