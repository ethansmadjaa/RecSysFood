from lib.recipes import get_recipes
import pandas as pd
from utils.logger import logger
import os

cache_path = "data/recipes_cache.json"
if os.path.exists(cache_path):
    try:
        already_cached_df = pd.read_json(cache_path)
        logger.info(f"Already cached df size: {len(already_cached_df)}")
    except ValueError as e:
        logger.warning(f"Cache file corrupted, will regenerate: {e}")
        already_cached_df = pd.DataFrame()
else:
    logger.info("No cache file found, starting fresh")
    already_cached_df = pd.DataFrame()
recipes = get_recipes()
df = pd.DataFrame([recipe.model_dump(mode="json") for recipe in recipes])
print(df.info())