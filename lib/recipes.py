import json
from pathlib import Path
from lib import supabase
from models.database import Recipe
from utils.logger import logger


def get_recipe(id: int) -> Recipe:
    logger.info(f"Fetching recipe with id {id}")
    response = supabase.table("recipes").select("*").eq("recipeid", id).execute()
    logger.info(f"Successfully fetched recipe {id}")
    return Recipe.model_validate(response.data[0])


def create_recipe(recipe: Recipe) -> Recipe:
    logger.info(f"Creating new recipe: {recipe.name}")
    response = supabase.table("recipes").insert(recipe.model_dump()).execute()
    logger.info(f"Successfully created recipe with id {response.data[0]}")
    return Recipe.model_validate(response.data[0])
