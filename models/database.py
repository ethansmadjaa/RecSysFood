"""Database models for RecSysFood.

Auto-generated from Supabase schema.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# Enums
class MealTypeEnum(str, Enum):
    """Meal type options."""

    BREAKFAST_BRUNCH = "breakfast_brunch"
    MAIN_COURSE = "main_course"
    STARTER_SIDE = "starter_side"
    DESSERT = "dessert"
    SNACK = "snack"


class CalorieGoalEnum(str, Enum):
    """Calorie goal options."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProteinGoalEnum(str, Enum):
    """Protein goal options."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Table Models
class User(BaseModel):
    """User model."""

    id: UUID
    auth_id: UUID
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    has_completed_signup: Optional[bool] = False
    created_at: datetime
    updated_at: datetime
    has_recommandations: Optional[bool] = False
    has_graded: Optional[bool] = False

    model_config = {"from_attributes": True}


class UserPreferences(BaseModel):
    """User preferences model."""

    user_preferences_id: int
    user_id: UUID
    meal_types: list[MealTypeEnum] = Field(default_factory=list)
    max_total_time: Optional[int] = None
    calorie_goal: CalorieGoalEnum = CalorieGoalEnum.MEDIUM
    protein_goal: ProteinGoalEnum = ProteinGoalEnum.MEDIUM
    dietary_restrictions: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class Recipe(BaseModel):
    """Recipe model."""

    recipeid: int
    name: str
    authorid: int
    authorname: str
    datepublished: datetime
    description: Optional[str] = None
    images: Optional[list[str]] = None
    recipecategory: Optional[str] = None
    keywords: Optional[list[str]] = None
    recipeingredientquantities: Optional[list[str]] = None
    recipeingredientparts: Optional[list[str]] = None
    aggregatedrating: Optional[float] = None
    reviewcount: Optional[float] = None
    calories: float
    fatcontent: float
    saturatedfatcontent: float
    cholesterolcontent: float
    sodiumcontent: float
    carbohydratecontent: float
    fibercontent: float
    sugarcontent: float
    proteincontent: float
    recipeservings: Optional[float] = None
    recipeyield: Optional[str] = None
    recipeinstructions: Optional[str] = None
    cooktime_min: int
    preptime_min: int
    totaltime_min: int
    is_vegan: bool
    is_vegetarian: bool
    contains_pork: bool
    contains_alcohol: bool
    contains_gluten: bool
    contains_nuts: bool
    contains_dairy: bool
    contains_egg: bool
    contains_fish: bool
    contains_soy: bool
    is_breakfast_brunch: bool
    is_dessert: bool
    calorie_category: str
    protein_category: str

    model_config = {"from_attributes": True}


class Interaction(BaseModel):
    """User-recipe interaction model."""

    interaction_id: int
    user_id: UUID
    recipe_id: int
    rating: Literal[0, 1, 2]
    created_at: datetime

    model_config = {"from_attributes": True}


# Insert Models (without auto-generated fields)
class UserInsert(BaseModel):
    """User insert model."""

    auth_id: UUID
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    has_completed_signup: Optional[bool] = False
    id: Optional[UUID] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class UserPreferencesInsert(BaseModel):
    """User preferences insert model."""

    user_id: UUID
    meal_types: list[MealTypeEnum] = Field(default_factory=list)
    max_total_time: Optional[int] = None
    calorie_goal: CalorieGoalEnum = CalorieGoalEnum.MEDIUM
    protein_goal: ProteinGoalEnum = ProteinGoalEnum.MEDIUM
    dietary_restrictions: list[dict[str, Any]] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class RecipeInsert(BaseModel):
    """Recipe insert model."""

    recipeid: int
    name: str
    authorid: int
    authorname: str
    datepublished: datetime
    description: Optional[str] = None
    images: Optional[list[str]] = None
    recipecategory: Optional[str] = None
    keywords: Optional[list[str]] = None
    recipeingredientquantities: Optional[list[str]] = None
    recipeingredientparts: Optional[list[str]] = None
    aggregatedrating: Optional[float] = None
    reviewcount: Optional[float] = None
    calories: float
    fatcontent: float
    saturatedfatcontent: float
    cholesterolcontent: float
    sodiumcontent: float
    carbohydratecontent: float
    fibercontent: float
    sugarcontent: float
    proteincontent: float
    recipeservings: Optional[float] = None
    recipeyield: Optional[str] = None
    recipeinstructions: Optional[str] = None
    cooktime_min: int
    preptime_min: int
    totaltime_min: int
    is_vegan: bool
    is_vegetarian: bool
    contains_pork: bool
    contains_alcohol: bool
    contains_gluten: bool
    contains_nuts: bool
    contains_dairy: bool
    contains_egg: bool
    contains_fish: bool
    contains_soy: bool
    is_breakfast_brunch: bool
    is_dessert: bool
    calorie_category: str
    protein_category: str


class InteractionInsert(BaseModel):
    """Interaction insert model."""

    user_id: UUID
    recipe_id: int
    rating: Literal[0, 1, 2]
    created_at: Optional[datetime] = None


# Update Models (all fields optional)
class UserUpdate(BaseModel):
    """User update model."""

    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    has_completed_signup: Optional[bool] = None
    updated_at: Optional[datetime] = None


class UserPreferencesUpdate(BaseModel):
    """User preferences update model."""

    meal_types: Optional[list[MealTypeEnum]] = None
    max_total_time: Optional[int] = None
    calorie_goal: Optional[CalorieGoalEnum] = None
    protein_goal: Optional[ProteinGoalEnum] = None
    dietary_restrictions: Optional[list[dict[str, Any]]] = None
    updated_at: Optional[datetime] = None


class RecipeUpdate(BaseModel):
    """Recipe update model."""

    name: Optional[str] = None
    authorid: Optional[int] = None
    authorname: Optional[str] = None
    datepublished: Optional[datetime] = None
    description: Optional[str] = None
    images: Optional[list[str]] = None
    recipecategory: Optional[str] = None
    keywords: Optional[list[str]] = None
    recipeingredientquantities: Optional[list[str]] = None
    recipeingredientparts: Optional[list[str]] = None
    aggregatedrating: Optional[float] = None
    reviewcount: Optional[float] = None
    calories: Optional[float] = None
    fatcontent: Optional[float] = None
    saturatedfatcontent: Optional[float] = None
    cholesterolcontent: Optional[float] = None
    sodiumcontent: Optional[float] = None
    carbohydratecontent: Optional[float] = None
    fibercontent: Optional[float] = None
    sugarcontent: Optional[float] = None
    proteincontent: Optional[float] = None
    recipeservings: Optional[float] = None
    recipeyield: Optional[str] = None
    recipeinstructions: Optional[str] = None
    cooktime_min: Optional[int] = None
    preptime_min: Optional[int] = None
    totaltime_min: Optional[int] = None
    is_vegan: Optional[bool] = None
    is_vegetarian: Optional[bool] = None
    contains_pork: Optional[bool] = None
    contains_alcohol: Optional[bool] = None
    contains_gluten: Optional[bool] = None
    contains_nuts: Optional[bool] = None
    contains_dairy: Optional[bool] = None
    contains_egg: Optional[bool] = None
    contains_fish: Optional[bool] = None
    contains_soy: Optional[bool] = None
    is_breakfast_brunch: Optional[bool] = None
    is_dessert: Optional[bool] = None
    calorie_category: Optional[str] = None
    protein_category: Optional[str] = None


class InteractionUpdate(BaseModel):
    """Interaction update model."""

    rating: Optional[Literal[0, 1, 2]] = None
