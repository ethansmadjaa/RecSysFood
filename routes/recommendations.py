from lib import supabase
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, field_validator
from typing import List, Literal, Optional, Any, cast
import pandas as pd
import ast
from utils.filtre_recommandation import UserPreferencesInput, select_recipes_from_preferences
from utils.recipes_loader import fetch_all_recipes
from utils.content_based_model import get_recommendations_for_user, model_exists
from utils.recsys_scheduler import train_and_save_model
from models.database import User, Interaction
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
    # Recommendation reason
    reason: Optional[str]

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

        prefs_data = cast(dict[str, Any], prefs_response.data)

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
        supabase.table("user_recommendations").delete().eq("user_id", user_id).eq("type", "filter").execute()

        # 6. Insert new recommendations
        for _, recipe in selected_recipes.iterrows():
            recommendation_data: dict[str, Any] = {
                "user_id": user_id,
                "recipe_id": int(recipe["recipeid"]),
                "score": float(recipe.get("score_total", 0)),
                "is_active": True,
                "type": "filter"
            }
            supabase.table("user_recommendations").insert(recommendation_data).execute()
        print(f"Generated {len(selected_recipes)} recommendations for user {user_id}")
    except Exception as e:
        print(f"Error generating recommendations for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

def update_user_has_recommandations(user_id: str):
    """Update user has_recommandations to True"""
    try:
        supabase.table("users").update({"has_recommandations": True}).eq(
            "id", user_id
        ).execute()
        return {"message": "User has_recommandations updated to True"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating user has_recommandations: {str(e)}")


def generate_recsys_recommendations_task(user_id: str):
    """Background task to generate content-based recommendations for a user.

    This uses the pre-trained model for fast inference. If the model doesn't exist,
    it will train the model first with the latest data.
    """
    try:
        print(f"[DEBUG] Generating recsys recommendations for user {user_id}")

        # Check if model exists
        if not model_exists():
            print(f"[DEBUG] Model doesn't exist. Training model...")

            # Train the model with latest data
            training_success = train_and_save_model()

            if not training_success:
                print(f"[ERROR] Model training failed for user {user_id}. Cannot generate recommendations.")
                return

        print(f"[DEBUG] Model exists, proceeding with recommendation generation")

        # Fetch user preferences
        print(f"[DEBUG] Fetching user preferences for user {user_id}")
        prefs_response = (
            supabase.table("user_preferences")
            .select("meal_types, calorie_goal, protein_goal")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )

        print(f"[DEBUG] Fetching user interactions for user {user_id}")
        interactions_response = (
            supabase.table("interactions")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )

        interactions_data = cast(list[dict[str, Any]], interactions_response.data or [])
        user_interactions = [Interaction(**i) for i in interactions_data]
        print(f"[DEBUG] Found {len(user_interactions)} interactions for user {user_id}")

        user_prefs = {}
        if prefs_response and prefs_response.data:
            user_prefs = cast(dict[str, Any], prefs_response.data)
            print(f"[DEBUG] User preferences: {user_prefs}")
        else:
            print(f"[DEBUG] No user preferences found for user {user_id}")

        # Get liked recipe IDs (rating > 0)
        liked_recipe_ids = [i["recipe_id"] for i in interactions_data]
        print(f"[DEBUG] Found {len(liked_recipe_ids)} liked recipes for user {user_id}")

        # Get recommendations from content-based model
        print(f"[DEBUG] Calling get_recommendations_for_user with k_user=500, k_recipe=100")
        recommendations = get_recommendations_for_user(
            user_id,
            top_k=15,
            user_prefs=user_prefs,
            liked_recipe_ids=liked_recipe_ids,
            user_interaction=user_interactions,
            k_user=500,
            k_recipe=100
        )

        if not recommendations:
            print(f"[WARNING] No recommendations generated for user {user_id}")
            return

        print(f"[DEBUG] Generated {len(recommendations)} recommendations for user {user_id}")

        # Clear old recsys recommendations for this user
        print(f"[DEBUG] Clearing old recsys recommendations for user {user_id}")
        supabase.table("user_recommendations").delete().eq("user_id", user_id).eq("type", "recsys").execute()

        # Insert new recommendations
        print(f"[DEBUG] Inserting {len(recommendations)} new recommendations")
        for idx, rec in enumerate(recommendations):
            recommendation_data = {
                "user_id": user_id,
                "recipe_id": rec["recipe_id"],
                "score": rec["score"],
                "is_active": True,
                "type": "recsys",
                "reason": rec.get("reason")
            }
            supabase.table("user_recommendations").insert(recommendation_data).execute()
            print(f"[DEBUG] Inserted recommendation {idx+1}/{len(recommendations)}: recipe_id={rec['recipe_id']}, score={rec['score']:.4f}")

        print(f"[SUCCESS] Generated {len(recommendations)} recsys recommendations for user {user_id}")

    except Exception as e:
        print(f"[ERROR] Error generating recsys recommendations for user {user_id}: {str(e)}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")


@router.post("/{user_id}/generate")
async def trigger_generate_recommendations(
    user_id: str, background_tasks: BackgroundTasks
):
    """Trigger recommendation generation in background"""
    # Verify user exists
    user_response = (
        supabase.table("users")
        .select("*")
        .eq("id", user_id)
        .maybe_single()
        .execute()
    )
    if user_response is None or not user_response.data:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Add task to background
    background_tasks.add_task(generate_recommendations_task, user_id)
    background_tasks.add_task(update_user_has_recommandations, user_id)
    return {"status": "generating", "message": "Recommendation generation started"}


@router.get("/{user_id}", response_model=RecommendationsResponse)
async def get_user_recommendations(
    user_id: str,
    type: Literal["filter", "recsys"] = Query(default="filter", description="Type of recommendations to fetch")
):
    """Get user's active recommendations with recipe details.

    Args:
        user_id: The user's UUID
        type: 'filter' for preference-filtered recipes (for grading),
              'recsys' for GraphSAGE personalized recommendations
    """
    try:
        # Get active recommendations for user
        recs_response = (
            supabase.table("user_recommendations")
            .select("recipe_id, score, reason")
            .eq("user_id", user_id)
            .eq("is_active", True)
            .eq("type", type)
            .execute()
        )

        if not recs_response.data:
            return RecommendationsResponse(status="not_found", recipes=[])

        # Get recipe IDs, scores, and reasons
        recs_data = cast(list[dict[str, Any]], recs_response.data)
        recipe_ids = [rec["recipe_id"] for rec in recs_data]
        scores_map = {rec["recipe_id"]: rec["score"] for rec in recs_data}
        reasons_map = {rec["recipe_id"]: rec.get("reason") for rec in recs_data}

        # Fetch recipe details
        recipes_response = (
            supabase.table("recipes").select("*").in_("recipeid", recipe_ids).execute()
        )

        if not recipes_response.data:
            return RecommendationsResponse(status="not_found", recipes=[])

        # Build response with scores and reasons
        recipes_data = cast(list[dict[str, Any]], recipes_response.data)
        recipes = []
        for recipe in recipes_data:
            try:
                # Add score and reason to recipe data
                recipe_data = {
                    **recipe,
                    "score": scores_map.get(recipe["recipeid"]),
                    "reason": reasons_map.get(recipe["recipeid"])
                }
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


@router.post("/{user_id}/more-recipes")
async def request_more_recipes_to_grade(
    user_id: str, background_tasks: BackgroundTasks
):
    """Request 20 more recipes to grade for refining the model.

    This generates new filter recommendations, excluding recipes the user has already rated.
    """
    try:
        # Verify user exists
        user_response = (
            supabase.table("users")
            .select("*")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )
        if user_response is None or not user_response.data:
            raise HTTPException(status_code=404, detail="User not found")

        # Get all recipe IDs the user has already rated
        interactions_response = (
            supabase.table("interactions")
            .select("recipe_id")
            .eq("user_id", user_id)
            .execute()
        )
        interactions_data = cast(list[dict[str, Any]], interactions_response.data or [])
        already_rated_ids = set(
            i["recipe_id"] for i in interactions_data
        )

        # Fetch user preferences
        prefs_response = (
            supabase.table("user_preferences")
            .select("*")
            .eq("user_id", user_id)
            .single()
            .execute()
        )

        if not prefs_response.data:
            raise HTTPException(status_code=404, detail="User preferences not found")

        prefs_data = cast(dict[str, Any], prefs_response.data)

        # Create UserPreferencesInput
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

        # Fetch all recipes
        recipes_df = fetch_all_recipes()

        if recipes_df.empty:
            raise HTTPException(status_code=404, detail="No recipes found")

        # Filter out already rated recipes
        recipes_df = recipes_df[~recipes_df["recipeid"].isin(already_rated_ids)]

        if recipes_df.empty:
            raise HTTPException(status_code=404, detail="No new recipes available to grade")

        # Generate recommendations (limited to 20)
        selected_recipes = select_recipes_from_preferences(recipes_df, prefs, n_display=20)

        # Clear old filter recommendations for this user
        supabase.table("user_recommendations").delete().eq("user_id", user_id).eq("type", "filter").execute()

        # Insert new recommendations
        for _, recipe in selected_recipes.iterrows():
            recommendation_data: dict[str, Any] = {
                "user_id": user_id,
                "recipe_id": int(recipe["recipeid"]),
                "score": float(recipe.get("score_total", 0)),
                "is_active": True,
                "type": "filter"
            }
            supabase.table("user_recommendations").insert(recommendation_data).execute()

        return {
            "status": "ready",
            "message": f"Generated {len(selected_recipes)} new recipes to grade",
            "count": len(selected_recipes)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating more recipes: {str(e)}"
        )


@router.delete("/{user_id}")
async def delete_user_recommendations(user_id: str):
    """Delete all recommendations for a user"""
    try:
        supabase.table("user_recommendations").delete().eq("user_id", user_id).eq("type", "filter").execute()
        return {"message": "Recommendations deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting recommendations: {str(e)}"
        )


@router.post("/{user_id}/regenerate-recsys")
async def regenerate_recsys_recommendations(
    user_id: str, background_tasks: BackgroundTasks
):
    """Regenerate GraphSAGE-based recommendations for a user.

    This deletes existing recsys recommendations and generates new ones.
    """
    try:
        # Verify user exists
        user_response = (
            supabase.table("users")
            .select("*")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )
        if user_response is None or not user_response.data:
            raise HTTPException(status_code=404, detail="User not found")

        # Delete existing recsys recommendations
        supabase.table("user_recommendations").delete().eq("user_id", user_id).eq("type", "recsys").execute()

        # Add task to generate new recommendations in background
        background_tasks.add_task(generate_recsys_recommendations_task, user_id)

        return {"status": "generating", "message": "Recsys recommendation regeneration started"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error regenerating recsys recommendations: {str(e)}"
        )
