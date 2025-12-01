from lib import supabase
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import  List
from models.database import Interaction
router = APIRouter(prefix="/api/interactions", tags=["interactions"])


class InteractionRequest(BaseModel):
    user_id: str
    recipe_id: int
    rating: int


class InteractionResponse(BaseModel):
    interaction_id: int
    user_id: str
    recipe_id: int
    rating: int
    created_at: str


@router.post("/", response_model=InteractionResponse)
async def create_interaction(interaction: InteractionRequest) -> InteractionResponse:
    try:
        interaction_response: Interaction = supabase.table("interactions").insert({
            "user_id": interaction.user_id,
            "recipe_id": interaction.recipe_id,
            "rating": interaction.rating
        }).execute()
        if not interaction_response.data:
            raise HTTPException(status_code=400, detail="Failed to create interaction")
        return InteractionResponse(**interaction_response.data[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating interaction: {str(e)}")


@router.get("/{user_id}", response_model=List[InteractionResponse])
async def get_interactions(user_id: str) -> list[InteractionResponse]:
    try:
        interactions_response = supabase.table("interactions").select("*").eq("user_id", user_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting interactions: {str(e)}")
    if not interactions_response.data:
        raise HTTPException(status_code=404, detail="No interactions found")
    return [InteractionResponse(**interaction) for interaction in interactions_response.data]


@router.patch("/{user_id}/complete-grading")
async def complete_grading(user_id: str):
    """Mark user as having completed grading all recommendations"""
    try:
        supabase.table("users").update({"has_graded": True}).eq("id", user_id).execute()
        return {"message": "Grading completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error completing grading: {str(e)}")