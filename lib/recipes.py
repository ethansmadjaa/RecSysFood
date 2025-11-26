import json
import os
from pathlib import Path
from lib import supabase
from models.database import Recipe
from utils.logger import logger

# Cache file path
CACHE_FILE = Path(__file__).parent.parent / "data" / "recipes_cache.json"
def get_recipes() -> list[Recipe]:
    """Get all recipes, using cache if available and incrementally updating it."""
    logger.info("Starting get_recipes()")
    
    # Load existing cache if available
    cached_recipes = []
    cached_ids = set()
    
    if CACHE_FILE.exists():
        logger.info(f"Cache file found at {CACHE_FILE}")
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                cached_recipes = [Recipe.model_validate(recipe) for recipe in cached_data]
                cached_ids = {recipe.recipeid for recipe in cached_recipes}
                logger.info(f"Loaded {len(cached_recipes)} recipes from cache")
        except (json.JSONDecodeError, Exception) as e:
            # If cache is corrupted, start fresh
            logger.warning(f"Failed to load cache: {e}. Starting fresh.")
    else:
        logger.info("No cache file found, will fetch all recipes from database")
    
    # Fetch from database with pagination
    all_recipes = []
    page_size = 1000
    offset = 0
    new_recipes_found = False
    
    logger.info("Starting database fetch with pagination")
    while True:
        logger.info(f"Fetching recipes from offset {offset} to {offset + page_size - 1}")
        response = supabase.table('recipes').select('*').range(offset, offset + page_size - 1).execute()
        
        if not response.data:
            logger.info("No more data to fetch from database")
            break
        
        logger.info(f"Fetched {len(response.data)} recipes in this batch")
        
        # Add new recipes and track if we found any
        for recipe_data in response.data:
            all_recipes.append(recipe_data)
            if recipe_data['recipeid'] not in cached_ids:
                new_recipes_found = True
        
        # Save incrementally after each batch if new recipes were found
        if new_recipes_found:
            logger.info(f"New recipes found, saving {len(all_recipes)} recipes to cache")
            try:
                CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                recipes_to_save = [Recipe.model_validate(recipe) for recipe in all_recipes]
                with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump([recipe.model_dump(mode="json") for recipe in recipes_to_save], f, ensure_ascii=False, indent=2)
                logger.info("Successfully saved recipes to cache")
                new_recipes_found = False  # Reset flag after saving
            except Exception as e:
                logger.error(f"Could not write recipes cache: {e}")
        
        # If we got fewer results than page_size, we've reached the end
        if len(response.data) < page_size:
            logger.info("Reached end of database results")
            break
            
        offset += page_size
    
    # If we fetched anything, return the fresh data
    if all_recipes:
        logger.info(f"Returning {len(all_recipes)} recipes from database fetch")
        recipes = [Recipe.model_validate(recipe) for recipe in all_recipes]
        
        # Final save to ensure everything is cached
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump([recipe.model_dump(mode="json") for recipe in recipes], f, ensure_ascii=False, indent=2)
            logger.info("Final cache save completed successfully")
        except Exception as e:
            logger.error(f"Could not write recipes cache: {e}")
        
        return recipes
    
    # If no data from database, return cached data
    logger.info(f"No data from database, returning {len(cached_recipes)} cached recipes")
    return cached_recipes

def get_recipe(id: int) -> Recipe:
    logger.info(f"Fetching recipe with id {id}")
    response = supabase.table('recipes').select('*').eq('recipeid', id).execute()
    logger.info(f"Successfully fetched recipe {id}")
    return Recipe.model_validate(response.data[0])

def create_recipe(recipe: Recipe) -> Recipe:
    logger.info(f"Creating new recipe: {recipe.name}")
    response = supabase.table('recipes').insert(recipe.model_dump()).execute()
    logger.info(f"Successfully created recipe with id {response.data[0]['recipeid']}")
    return Recipe.model_validate(response.data[0])
