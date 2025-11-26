from lib.supabase import supabase
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

from filtre_recommandation import UserPreferencesInput, select_recipes_from_preferences

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


class RecipeResponse(BaseModel):
    recipeid: int
    name: str
    images: Optional[List[str]]
    totaltime_min: Optional[int]
    aggregatedrating: Optional[float]
    reviewcount: Optional[float]
    calories: Optional[float]
    proteincontent: Optional[float]
    is_vegan: Optional[bool]
    is_vegetarian: Optional[bool]
    score: Optional[float]


class RecommendationsResponse(BaseModel):
    status: str  # 'ready', 'generating', 'not_found'
    recipes: List[RecipeResponse]


def fetch_all_recipes() -> pd.DataFrame:
    """Fetch all recipes from Supabase"""
    response = supabase.table('recipes').select('*').execute()
    if not response.data:
        return pd.DataFrame()
    return pd.DataFrame(response.data)


def generate_recommendations_task(user_id: str):
    """Background task to generate recommendations for a user"""
    try:
        # 1. Fetch user preferences
        prefs_response = supabase.table('user_preferences').select('*').eq('user_id', user_id).single().execute()

        if not prefs_response.data:
            print(f"No preferences found for user {user_id}")
            return

        prefs_data = prefs_response.data

        # 2. Create UserPreferencesInput from database data
        prefs = UserPreferencesInput(
            meal_types=prefs_data.get('meal_types', []),
            max_total_time=prefs_data.get('max_total_time'),
            calorie_goal=prefs_data.get('calorie_goal', 'medium'),
            protein_goal=prefs_data.get('protein_goal', 'medium'),
            dietary_restrictions=prefs_data.get('dietary_restrictions', []),
            allergy_nuts=prefs_data.get('allergy_nuts', False),
            allergy_dairy=prefs_data.get('allergy_dairy', False),
            allergy_egg=prefs_data.get('allergy_egg', False),
            allergy_fish=prefs_data.get('allergy_fish', False),
            allergy_soy=prefs_data.get('allergy_soy', False),
        )

        # 3. Fetch all recipes
        recipes_df = fetch_all_recipes()

        if recipes_df.empty:
            print("No recipes found in database")
            return

        # 4. Generate recommendations
        selected_recipes = select_recipes_from_preferences(recipes_df, prefs)

        # 5. Clear old recommendations for this user
        supabase.table('user_recommendations').delete().eq('user_id', user_id).execute()

        # 6. Insert new recommendations
        for _, recipe in selected_recipes.iterrows():
            recommendation_data = {
                "user_id": user_id,
                "recipe_id": int(recipe['recipeid']),
                "score": float(recipe.get('score_total', 0)),
                "is_active": True
            }
            supabase.table('user_recommendations').insert(recommendation_data).execute()

        print(f"Generated {len(selected_recipes)} recommendations for user {user_id}")

    except Exception as e:
        print(f"Error generating recommendations for user {user_id}: {str(e)}")


@router.post("/{user_id}/generate")
async def trigger_generate_recommendations(user_id: str, background_tasks: BackgroundTasks):
    """Trigger recommendation generation in background"""
    # Verify user exists
    user_response = supabase.table('users').select('id').eq('id', user_id).single().execute()
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
        recs_response = supabase.table('user_recommendations').select(
            'recipe_id, score'
        ).eq('user_id', user_id).eq('is_active', True).execute()

        if not recs_response.data:
            return RecommendationsResponse(status="not_found", recipes=[])

        # Get recipe IDs
        recipe_ids = [rec['recipe_id'] for rec in recs_response.data]
        scores_map = {rec['recipe_id']: rec['score'] for rec in recs_response.data}

        # Fetch recipe details
        recipes_response = supabase.table('recipes').select(
            'recipeid, name, images, totaltime_min, aggregatedrating, reviewcount, calories, proteincontent, is_vegan, is_vegetarian'
        ).in_('recipeid', recipe_ids).execute()

        if not recipes_response.data:
            return RecommendationsResponse(status="not_found", recipes=[])

        # Build response with scores
        recipes = []
        for recipe in recipes_response.data:
            recipes.append(RecipeResponse(
                recipeid=recipe['recipeid'],
                name=recipe['name'],
                images=recipe.get('images'),
                totaltime_min=recipe.get('totaltime_min'),
                aggregatedrating=recipe.get('aggregatedrating'),
                reviewcount=recipe.get('reviewcount'),
                calories=recipe.get('calories'),
                proteincontent=recipe.get('proteincontent'),
                is_vegan=recipe.get('is_vegan'),
                is_vegetarian=recipe.get('is_vegetarian'),
                score=scores_map.get(recipe['recipeid'])
            ))

        # Sort by score descending
        recipes.sort(key=lambda x: x.score or 0, reverse=True)

        return RecommendationsResponse(status="ready", recipes=recipes)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching recommendations: {str(e)}")


@router.delete("/{user_id}")
async def delete_user_recommendations(user_id: str):
    """Delete all recommendations for a user"""
    try:
        supabase.table('user_recommendations').delete().eq('user_id', user_id).execute()
        return {"message": "Recommendations deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting recommendations: {str(e)}")
