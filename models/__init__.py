"""Database models package."""

from models.database import (
    CalorieGoalEnum,
    Interaction,
    InteractionInsert,
    InteractionUpdate,
    MealTypeEnum,
    ProteinGoalEnum,
    Recipe,
    RecipeInsert,
    RecipeUpdate,
    User,
    UserInsert,
    UserPreferences,
    UserPreferencesInsert,
    UserPreferencesUpdate,
    UserUpdate,
)

__all__ = [
    # Enums
    "MealTypeEnum",
    "CalorieGoalEnum",
    "ProteinGoalEnum",
    # Models
    "User",
    "UserPreferences",
    "Recipe",
    "Interaction",
    # Insert models
    "UserInsert",
    "UserPreferencesInsert",
    "RecipeInsert",
    "InteractionInsert",
    # Update models
    "UserUpdate",
    "UserPreferencesUpdate",
    "RecipeUpdate",
    "InteractionUpdate",
]
