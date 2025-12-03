"""Scheduler for periodic GraphSAGE model training.

This module handles hourly retraining of the recommendation model
using the latest data from Supabase.
"""

import threading
import time
from datetime import datetime
import os

import pandas as pd
import schedule
from dotenv import load_dotenv

from lib import supabase
from utils.content_based_model import Train_GNN, save_model, model_exists

load_dotenv()

# Global lock to prevent concurrent training
_training_lock = threading.Lock()
_scheduler_thread = None
_is_running = False


def fetch_training_data():
    """Fetch all required data from Supabase for model training.

    Returns:
        Tuple of (user_preferences_df, recipes_df, interactions_df) or None if error
    """
    try:
        # Fetch user preferences
        prefs_response = supabase.table("user_preferences").select(
            "user_id, meal_types, calorie_goal, protein_goal"
        ).execute()

        if not prefs_response.data:
            print("No user preferences found in database")
            return None

        user_preferences_df = pd.DataFrame(prefs_response.data)

        recipes_df = pd.read_csv(os.getenv("CSV_PATH"))

        # Fetch interactions
        interactions_response = supabase.table("interactions").select(
            "interaction_id, user_id, recipe_id, rating"
        ).execute()

        if not interactions_response.data:
            print("No interactions found in database. Model training requires user ratings.")
            return None

        interactions_df = pd.DataFrame(interactions_response.data)

        print(f"Fetched {len(user_preferences_df)} users, {len(recipes_df)} recipes, {len(interactions_df)} interactions")

        return user_preferences_df, recipes_df, interactions_df

    except Exception as e:
        print(f"Error fetching training data: {e}")
        return None


def train_and_save_model():
    """Train the GraphSAGE model with latest data and save to disk.

    This function is thread-safe and uses a lock to prevent concurrent training.
    """
    if not _training_lock.acquire(blocking=False):
        print("Training already in progress, skipping...")
        return False

    try:
        print(f"[{datetime.now().isoformat()}] Starting GraphSAGE model training...")

        # Fetch data
        training_data = fetch_training_data()
        print("training_data fetched")
        if training_data is None:
            print("Could not fetch training data. Skipping training.")
            return False

        user_preferences_df, recipes_df, interactions_df = training_data

        # Check minimum data requirements
        if len(interactions_df) < 10:
            print(f"Insufficient interactions ({len(interactions_df)}). Need at least 10 for training.")
            return False
        else:
            print("training_data fetched and checked successfully")

        # Train model
        recipe_embeddings, model, recipe_id_to_idx, idx_to_recipe_id, data = Train_GNN(
            user_preferences_df,
            recipes_df,
            interactions_df,
            epochs=50
        )

        if model is None:
            print("Model training returned None. Skipping save.")
            return False

        # Save model
        save_model(model, data, recipes_df, recipe_id_to_idx, idx_to_recipe_id)

        print(f"[{datetime.now().isoformat()}] Model training completed successfully")
        return True

    except Exception as e:
        print(f"Error during model training: {e}")
        return False

    finally:
        _training_lock.release()


def _run_scheduler():
    """Background thread function to run the scheduler."""
    global _is_running
    while _is_running:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


def start_scheduler():
    """Start the background scheduler for hourly model training.

    This starts a daemon thread that runs the scheduler.
    """
    global _scheduler_thread, _is_running

    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        print("Scheduler already running")
        return

    # Schedule hourly training
    schedule.clear()
    schedule.every(1).hour.do(train_and_save_model)

    _is_running = True
    _scheduler_thread = threading.Thread(target=_run_scheduler, daemon=True)
    _scheduler_thread.start()

    print("Started hourly model training scheduler")


def stop_scheduler():
    """Stop the background scheduler."""
    global _is_running
    _is_running = False
    schedule.clear()
    print("Stopped model training scheduler")


def initialize_model_if_needed():
    """Initialize the model on startup if it doesn't exist.

    This is called during application startup to ensure a model exists
    for inference.
    """
    if model_exists():
        print("Pre-trained model found. Ready for inference.")
        return True

    print("No pre-trained model found. Attempting initial training...")
    return train_and_save_model()
