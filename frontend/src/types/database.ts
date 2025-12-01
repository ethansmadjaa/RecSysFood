// Database Types
// Auto-generated from Supabase schema

// Enums
export type MealTypeEnum =
  | 'breakfast_brunch'
  | 'main_course'
  | 'starter_side'
  | 'dessert'
  | 'snack';

export type CalorieGoalEnum = 'low' | 'medium' | 'high';

export type ProtegeGoalEnum = 'low' | 'medium' | 'high';

// Dietary Restriction Type (stored as JSONB)
export interface DietaryRestriction {
  name: string;
  [key: string]: unknown;
}

// Tables
export interface User {
  id: string;
  auth_id: string;
  email: string;
  first_name: string | null;
  last_name: string | null;
  has_completed_signup: boolean | null;
  created_at: string;
  updated_at: string;
  has_recommandations: boolean;
  has_graded: boolean;
}

export interface UserPreferences {
  user_preferences_id: number;
  user_id: string;
  meal_types: MealTypeEnum[];
  max_total_time: number | null;
  calorie_goal: CalorieGoalEnum;
  protein_goal: ProtegeGoalEnum;
  dietary_restrictions: DietaryRestriction[];
  created_at: string;
  updated_at: string;
}

export interface Recipe {
  recipeid: number;
  name: string;
  authorid: number;
  authorname: string;
  datepublished: string;
  description: string | null;
  images: string[] | null;
  recipecategory: string | null;
  keywords: string[] | null;
  recipeingredientquantities: string[] | null;
  recipeingredientparts: string[] | null;
  aggregatedrating: number | null;
  reviewcount: number | null;
  calories: number;
  fatcontent: number;
  saturatedfatcontent: number;
  cholesterolcontent: number;
  sodiumcontent: number;
  carbohydratecontent: number;
  fibercontent: number;
  sugarcontent: number;
  proteincontent: number;
  recipeservings: number | null;
  recipeyield: string | null;
  recipeinstructions: string | null;
  cooktime_min: number;
  preptime_min: number;
  totaltime_min: number;
  is_vegan: boolean;
  is_vegetarian: boolean;
  contains_pork: boolean;
  contains_alcohol: boolean;
  contains_gluten: boolean;
  contains_nuts: boolean;
  contains_dairy: boolean;
  contains_egg: boolean;
  contains_fish: boolean;
  contains_soy: boolean;
  is_breakfast_brunch: boolean;
  is_dessert: boolean;
  calorie_category: string;
  protein_category: string;
}

export interface Interaction {
  interaction_id: number;
  user_id: string;
  recipe_id: number;
  rating: 0 | 1 | 2;
  created_at: string;
}

// Insert Types (without auto-generated fields)
export type UserInsert = Omit<User, 'id' | 'created_at' | 'updated_at'> & {
  id?: string;
  created_at?: string;
  updated_at?: string;
  has_completed_signup?: boolean | null;
  has_recommandations?: boolean;
  has_graded?: boolean;
};

export type UserPreferencesInsert = Omit<UserPreferences, 'user_preferences_id' | 'created_at' | 'updated_at'> & {
  created_at?: string;
  updated_at?: string;
  meal_types?: MealTypeEnum[];
  max_total_time?: number | null;
  calorie_goal?: CalorieGoalEnum;
  protein_goal?: ProtegeGoalEnum;
  dietary_restrictions?: DietaryRestriction[];
};

export type RecipeInsert = Recipe;

export type InteractionInsert = Omit<Interaction, 'interaction_id' | 'created_at'> & {
  created_at?: string;
};

// Update Types (all fields optional except ID)
export type UserUpdate = Partial<Omit<User, 'id' | 'auth_id' | 'created_at'>> & {
  updated_at?: string;
};

export type UserPreferencesUpdate = Partial<Omit<UserPreferences, 'user_preferences_id' | 'user_id' | 'created_at'>> & {
  updated_at?: string;
};

export type RecipeUpdate = Partial<Omit<Recipe, 'recipeid'>>;

export type InteractionUpdate = Partial<Omit<Interaction, 'interaction_id' | 'user_id' | 'recipe_id'>>;

// Database type combining all tables
export interface Database {
  public: {
    Tables: {
      users: {
        Row: User;
        Insert: UserInsert;
        Update: UserUpdate;
      };
      user_preferences: {
        Row: UserPreferences;
        Insert: UserPreferencesInsert;
        Update: UserPreferencesUpdate;
      };
      recipes: {
        Row: Recipe;
        Insert: RecipeInsert;
        Update: RecipeUpdate;
      };
      interactions: {
        Row: Interaction;
        Insert: InteractionInsert;
        Update: InteractionUpdate;
      };
    };
    Enums: {
      meal_type_enum: MealTypeEnum;
      calorie_goal_enum: CalorieGoalEnum;
      protein_goal_enum: ProtegeGoalEnum;
    };
  };
}
