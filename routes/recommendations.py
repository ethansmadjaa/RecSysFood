from lib import supabase
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, field_validator
from typing import List, Optional
import pandas as pd
import ast

from filtre_recommandation import UserPreferencesInput, select_recipes_from_preferences
from utils.recipes_loader import fetch_all_recipes

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


def parse_string_list(value) -> Optional[List[str]]:
    """Parse a string representation of a list into an actual list."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        # Try to parse string representation of a list
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        # If it's a plain string, return as single-item list
        return [value] if value.strip() else None
    return None


class RecipeResponse(BaseModel):
    recipeid: int
    name: str
    description: Optional[str]
    authorname: Optional[str]
    datepublished: Optional[str]
    images: Optional[List[str]]
    recipecategory: Optional[str]
    keywords: Optional[List[str]]
    recipeingredientquantities: Optional[List[str]]
    recipeingredientparts: Optional[List[str]]
    recipeinstructions: Optional[List[str]]
    cooktime_min: Optional[int]
    preptime_min: Optional[int]
    totaltime_min: Optional[int]
    recipeservings: Optional[int]
    recipeyield: Optional[str]
    aggregatedrating: Optional[float]
    reviewcount: Optional[float]
    # Nutrition
    calories: Optional[float]
    fatcontent: Optional[float]
    saturatedfatcontent: Optional[float]
    cholesterolcontent: Optional[float]
    sodiumcontent: Optional[float]
    carbohydratecontent: Optional[float]
    fibercontent: Optional[float]
    sugarcontent: Optional[float]
    proteincontent: Optional[float]
    # Dietary flags
    is_vegan: Optional[bool]
    is_vegetarian: Optional[bool]
    contains_pork: Optional[bool]
    contains_alcohol: Optional[bool]
    contains_gluten: Optional[bool]
    contains_nuts: Optional[bool]
    contains_dairy: Optional[bool]
    contains_egg: Optional[bool]
    contains_fish: Optional[bool]
    contains_soy: Optional[bool]
    # Meal type flags
    is_breakfast_brunch: Optional[bool]
    is_dessert: Optional[bool]
    # Categories
    calorie_category: Optional[str]
    protein_category: Optional[str]
    # Recommendation score
    score: Optional[float]

    # Validators to handle string representations of lists from the database
    @field_validator(
        "images",
        "keywords",
        "recipeingredientquantities",
        "recipeingredientparts",
        "recipeinstructions",
        mode="before",
    )
    @classmethod
    def parse_list_fields(cls, v):
        return parse_string_list(v)


class RecommendationsResponse(BaseModel):
    status: str  # 'ready', 'generating', 'not_found'
    recipes: List[RecipeResponse]




def generate_recommendations_task(user_id: str):
    """Background task to generate recommendations for a user"""
    try:
        # 1. Fetch user preferences
        prefs_response = (
            supabase.table("user_preferences")
            .select("*")
            .eq("user_id", user_id)
            .single()
            .execute()
        )

        if not prefs_response.data:
            print(f"No preferences found for user {user_id}")
            return

        prefs_data = prefs_response.data

        # 2. Create UserPreferencesInput from database data
        prefs = UserPreferencesInput(
            meal_types=prefs_data.get("meal_types", []),
            max_total_time=prefs_data.get("max_total_time"),
            calorie_goal=prefs_data.get("calorie_goal", "medium"),
            protein_goal=prefs_data.get("protein_goal", "medium"),
            dietary_restrictions=prefs_data.get("dietary_restrictions", []),
            allergy_nuts=prefs_data.get("allergy_nuts", False),
            allergy_dairy=prefs_data.get("allergy_dairy", False),
            allergy_egg=prefs_data.get("allergy_egg", False),
            allergy_fish=prefs_data.get("allergy_fish", False),
            allergy_soy=prefs_data.get("allergy_soy", False),
        )

        # 3. Fetch all recipes
        recipes_df = fetch_all_recipes()

        if recipes_df.empty:
            print("No recipes found in database")
            return

        # 4. Generate recommendations
        selected_recipes = select_recipes_from_preferences(recipes_df, prefs)

        # 5. Clear old recommendations for this user
        supabase.table("user_recommendations").delete().eq("user_id", user_id).execute()

        # 6. Insert new recommendations
        for _, recipe in selected_recipes.iterrows():
            recommendation_data = {
                "user_id": user_id,
                "recipe_id": int(recipe["recipeid"]),
                "score": float(recipe.get("score_total", 0)),
                "is_active": True,
            }
            supabase.table("user_recommendations").insert(recommendation_data).execute()

        print(f"Generated {len(selected_recipes)} recommendations for user {user_id}")

    except Exception as e:
        print(f"Error generating recommendations for user {user_id}: {str(e)}")


@router.post("/{user_id}/generate")
async def trigger_generate_recommendations(
    user_id: str, background_tasks: BackgroundTasks
):
    """Trigger recommendation generation in background"""
    # Verify user exists
    user_response = (
        supabase.table("users").select("id").eq("id", user_id).single().execute()
    )
    if not user_response.data:
        raise HTTPException(status_code=404, detail="User not found")

    # Add task to background
    background_tasks.add_task(generate_recommendations_task, user_id)

    return {"status": "generating", "message": "Recommendation generation started"}

@router.get("/{user_id}", response_model=RecommendationsResponse)
async def get_user_recommendations(user_id: str):
    """Get user's active recommendations with recipe details"""
    try:
        # Get active recommendations for user
        recs_response = (
            supabase.table("user_recommendations")
            .select("recipe_id, score")
            .eq("user_id", user_id)
            .eq("is_active", True)
            .execute()
        )

        if not recs_response.data:
            return RecommendationsResponse(status="not_found", recipes=[])

        # Get recipe IDs and scores
        recipe_ids = [rec["recipe_id"] for rec in recs_response.data]
        scores_map = {rec["recipe_id"]: rec["score"] for rec in recs_response.data}

        # Fetch recipe details
        recipes_response = (
            supabase.table("recipes")
            .select("*")
            .in_("recipeid", recipe_ids)
            .execute()
        )

        if not recipes_response.data:
            return RecommendationsResponse(status="not_found", recipes=[])

        # Build response with scores
        recipes = []
        for recipe in recipes_response.data:
            try:
                # Add score to recipe data and pass as dict for validators to process
                recipe_data = {**recipe, "score": scores_map.get(recipe["recipeid"])}
                recipe_obj = RecipeResponse.model_validate(recipe_data)
                recipes.append(recipe_obj)
            except Exception:
                # Continue processing other recipes instead of failing completely
                continue

        # Sort by score descending
        recipes.sort(key=lambda x: x.score or 0, reverse=True)

        return RecommendationsResponse(status="ready", recipes=recipes)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching recommendations: {str(e)}"
        )

@router.delete("/{user_id}")
async def delete_user_recommendations(user_id: str):
    """Delete all recommendations for a user"""
    try:
        supabase.table("user_recommendations").delete().eq("user_id", user_id).execute()
        return {"message": "Recommendations deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting recommendations: {str(e)}"
        )
