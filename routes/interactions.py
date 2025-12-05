from lib import supabase
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from models.database import Interaction
from routes.recommendations import generate_recsys_recommendations_task

router = APIRouter(prefix="/api/interactions", tags=["interactions"])


class InteractionRequest(BaseModel):
    user_id: str
    recipe_id: int
    rating: int


class InteractionResponse(BaseModel):
    interaction_id: int
    user_id: UUID
    recipe_id: int
    rating: int
    created_at: datetime


@router.post("/", response_model=InteractionResponse)
async def create_interaction(interaction: InteractionRequest):
    try:
        response = supabase.table("interactions").insert({
            "user_id": interaction.user_id,
            "recipe_id": interaction.recipe_id,
            "rating": interaction.rating
        }).execute()
        if not response.data:
            raise HTTPException(status_code=400, detail="Failed to create interaction")
        return InteractionResponse.model_validate(response.data[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating interaction: {str(e)}")


@router.get("/{user_id}", response_model=Optional[List[InteractionResponse]])
async def get_interactions(user_id: str):
    try:
        response = supabase.table("interactions").select("*").eq("user_id", user_id).execute()
        interactions: list[Interaction] = [Interaction.model_validate(interaction) for interaction in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting interactions: {str(e)}")
    return [InteractionResponse.model_validate(interaction.model_dump()) for interaction in interactions] if interactions else None


@router.patch("/{user_id}/complete-grading")
async def complete_grading(user_id: str, background_tasks: BackgroundTasks):
    """Mark user as having completed grading and trigger recsys recommendations.

    This endpoint:
    1. Marks the user as having completed grading (has_graded = True)
    2. Triggers content-based recommendation generation in background
    """
    try:
        supabase.table("users").update({"has_graded": True}).eq("id", user_id).execute()

        # Trigger recsys recommendation generation in background
        background_tasks.add_task(generate_recsys_recommendations_task, user_id)

        return {"message": "Grading completed successfully. Generating personalized recommendations..."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error completing grading: {str(e)}")