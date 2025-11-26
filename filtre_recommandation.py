from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np
from utils.recipes_loader import fetch_all_recipes
from pydantic import BaseModel

# ============================
# 1. Modèle des préférences utilisateur (aligné sur le frontend)
# ============================


class UserPreferencesInput(BaseModel):
    """
    Représente les préférences envoyées par le frontend.
    Format aligné sur les types TypeScript du frontend.

    meal_types: ['breakfast_brunch', 'main_course', 'starter_side', 'dessert', 'snack']
    max_total_time: 15, 30, 45, 60, ou None
    calorie_goal: 'low', 'medium', 'high'
    protein_goal: 'low', 'medium', 'high'
    dietary_restrictions: ['vegetarian', 'vegan', 'no_pork', 'no_alcohol', 'gluten_free']
    """

    meal_types: List[str]
    max_total_time: Optional[int]
    calorie_goal: str
    protein_goal: str
    dietary_restrictions: List[str]
    allergy_nuts: bool
    allergy_dairy: bool
    allergy_egg: bool
    allergy_fish: bool
    allergy_soy: bool


# ============================
# 2. Helpers
# ============================


def build_meal_type_mask(df: pd.DataFrame, meal_types: List[str]) -> pd.Series:
    """
    Construit un masque basé sur les types de repas sélectionnés par l'utilisateur.

    Frontend values -> Database columns:
    - 'breakfast_brunch' -> is_breakfast_brunch == True
    - 'dessert' -> is_dessert == True
    - 'snack' -> is_dessert == True (assimilé)
    - 'main_course' -> ni breakfast ni dessert
    - 'starter_side' -> ni breakfast ni dessert (assimilé à main)
    """
    if not meal_types:
        return pd.Series(True, index=df.index)

    mask = pd.Series(False, index=df.index)

    if "breakfast_brunch" in meal_types and "is_breakfast_brunch" in df.columns:
        mask |= df["is_breakfast_brunch"]

    if "dessert" in meal_types and "is_dessert" in df.columns:
        mask |= df["is_dessert"]

    if "snack" in meal_types and "is_dessert" in df.columns:
        mask |= df["is_dessert"]

    if "main_course" in meal_types or "starter_side" in meal_types:
        if {"is_breakfast_brunch", "is_dessert"}.issubset(df.columns):
            mask |= ~df["is_breakfast_brunch"] & ~df["is_dessert"]

    return mask


def map_goal_to_category(goal: str) -> Optional[str]:
    """
    Convertit les valeurs frontend (lowercase) vers les valeurs de la DB (capitalized).
    'low' -> 'Low', 'medium' -> 'Medium', 'high' -> 'High'
    """
    mapping = {"low": "Low", "medium": "Medium", "high": "High"}
    return mapping.get(goal)


def safe_norm(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.min() == s.max():
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


# ============================
# 3. Filtrage à partir des préférences
# ============================


def filter_recipes_with_preferences(
    df: pd.DataFrame, prefs: UserPreferencesInput
) -> pd.DataFrame:
    recipes = df.copy()

    # --- Type de plat / moment ---
    meal_mask = build_meal_type_mask(recipes, prefs.meal_types)
    recipes = recipes[meal_mask]

    # --- Temps max ---
    if prefs.max_total_time is not None and "totaltime_min" in recipes.columns:
        recipes = recipes[recipes["totaltime_min"] <= prefs.max_total_time]

    # --- Objectif calories (Calorie_Category) ---
    calorie_cat = map_goal_to_category(prefs.calorie_goal)
    if calorie_cat and "calorie_category" in recipes.columns:
        recipes = recipes[recipes["calorie_category"] == calorie_cat]

    # --- Objectif protéines (Protein_Category) ---
    protein_cat = map_goal_to_category(prefs.protein_goal)
    if protein_cat and "protein_category" in recipes.columns:
        recipes = recipes[recipes["protein_category"] == protein_cat]

    # --- Restrictions alimentaires ---
    if (
        "vegetarian" in prefs.dietary_restrictions
        and "is_vegetarian" in recipes.columns
    ):
        recipes = recipes[recipes["is_vegetarian"]]

    if "vegan" in prefs.dietary_restrictions and "is_vegan" in recipes.columns:
        recipes = recipes[recipes["is_vegan"]]

    if "no_pork" in prefs.dietary_restrictions and "contains_pork" in recipes.columns:
        recipes = recipes[~recipes["contains_pork"]]

    if (
        "no_alcohol" in prefs.dietary_restrictions
        and "contains_alcohol" in recipes.columns
    ):
        recipes = recipes[~recipes["contains_alcohol"]]

    if (
        "gluten_free" in prefs.dietary_restrictions
        and "contains_gluten" in recipes.columns
    ):
        recipes = recipes[~recipes["contains_gluten"]]

    # --- Allergies ---
    if prefs.allergy_nuts and "contains_nuts" in recipes.columns:
        recipes = recipes[~recipes["contains_nuts"]]

    if prefs.allergy_dairy and "contains_dairy" in recipes.columns:
        recipes = recipes[~recipes["contains_dairy"]]

    if prefs.allergy_egg and "contains_egg" in recipes.columns:
        recipes = recipes[~recipes["contains_egg"]]

    if prefs.allergy_fish and "contains_fish" in recipes.columns:
        recipes = recipes[~recipes["contains_fish"]]

    if prefs.allergy_soy and "contains_soy" in recipes.columns:
        recipes = recipes[~recipes["contains_soy"]]

    return recipes


# ============================
# 4. Scoring (qualité + nutrition)
# ============================


def add_scores_with_preferences(
    df: pd.DataFrame, prefs: UserPreferencesInput
) -> pd.DataFrame:
    recipes = df.copy()

    # --- Quality score : AggregatedRating + ReviewCount ---
    if "aggregatedrating" in recipes.columns:
        rating = recipes["aggregatedrating"].fillna(
            recipes["aggregatedrating"].median()
        )
    else:
        rating = pd.Series(0.0, index=recipes.index)

    if "reviewcount" in recipes.columns:
        reviews = recipes["reviewcount"].fillna(0)
    else:
        reviews = pd.Series(0.0, index=recipes.index)

    rating_norm = safe_norm(rating)
    reviews_norm = safe_norm(np.log1p(reviews))

    recipes["quality_score"] = 0.7 * rating_norm + 0.3 * reviews_norm

    # --- Nutrition score : basé sur Calories + ProteinContent + objectifs ---
    calories = recipes["calories"].astype(float).fillna(recipes["calories"].median())
    proteins = (
        recipes["proteincontent"]
        .astype(float)
        .fillna(recipes["proteincontent"].median())
    )

    # Score calories : par défaut neutre
    cal_score = pd.Series(0.5, index=recipes.index)
    if prefs.calorie_goal == "low":
        cal_score = safe_norm(-calories)  # moins = mieux
    elif prefs.calorie_goal == "high":
        cal_score = safe_norm(calories)  # plus = mieux

    # Score protéines
    prot_score = pd.Series(0.5, index=recipes.index)
    if prefs.protein_goal == "high":
        prot_score = safe_norm(proteins)  # plus = mieux
    elif prefs.protein_goal == "low":
        prot_score = safe_norm(-proteins)  # moins = mieux

    recipes["nutrition_score"] = 0.5 * cal_score + 0.5 * prot_score

    # --- Score total ---
    w_quality = 0.6
    w_nutrition = 0.4
    recipes["score_total"] = (
        w_quality * recipes["quality_score"] + w_nutrition * recipes["nutrition_score"]
    )

    return recipes


# ============================
# 5. Algorithme complet : sélectionner 15 recettes
# ============================


def select_recipes_from_preferences(
    recipes_df: pd.DataFrame,
    prefs: UserPreferencesInput,
    n_display: int = 15,
    pool_size: int = 100,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    1) Filtre les recettes avec les préférences utilisateur
    2) Calcule un score qualité + nutrition
    3) Trie par score_total
    4) Garde un pool (Top pool_size)
    5) Tire n_display recettes au hasard dans ce pool
    """

    filtered = filter_recipes_with_preferences(recipes_df, prefs)

    # fallback si filtres trop stricts
    if filtered.empty:
        filtered = recipes_df.copy()

    scored = add_scores_with_preferences(filtered, prefs)
    scored = scored.sort_values("score_total", ascending=False)

    pool = scored.head(min(pool_size, len(scored)))
    n_to_sample = min(n_display, len(pool))

    selected = pool.sample(n=n_to_sample, random_state=random_state)

    cols_for_front = [
        "recipeid",
        "name",
        "images",
        "totaltime_min",
        "aggregatedrating",
        "reviewcount",
        "calories",
        "proteincontent",
        "is_vegan",
        "is_vegetarian",
        "is_breakfast_brunch",
        "is_dessert",
        "calorie_category",
        "protein_category",
        "score_total",
    ]
    cols_for_front = [c for c in cols_for_front if c in selected.columns]

    return selected[cols_for_front]


if __name__ == "__main__":
    recipes = fetch_all_recipes()       
    df = pd.DataFrame([recipe.model_dump(mode="json") for recipe in recipes])

    prefs = UserPreferencesInput(
        meal_types=["main_course", "dessert"],
        max_total_time=30,
        calorie_goal="low",
        protein_goal="high",
        dietary_restrictions=["no_pork", "no_alcohol"],
        allergy_nuts=True,
        allergy_dairy=False,
        allergy_egg=False,
        allergy_fish=False,
        allergy_soy=False,
    )

    selected = select_recipes_from_preferences(df, prefs)
    print(selected.head())
