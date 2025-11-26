import { supabase } from '@/lib/supabase/client'
import type { Recipe } from './recommendations'

export interface Favorite {
  id: number
  user_id: string
  recipe_id: number
  created_at: string
}

export interface FavoriteWithRecipe extends Favorite {
  recipes: Recipe
}

/**
 * Get all favorites for a user with full recipe data
 */
export async function getUserFavorites(userId: string): Promise<{ data: Recipe[] | null; error: string | null }> {
  try {
    const { data, error } = await supabase
      .from('favorites')
      .select(`
        id,
        recipe_id,
        created_at,
        recipes (*)
      `)
      .eq('user_id', userId)
      .order('created_at', { ascending: false })

    if (error) {
      console.error('Error fetching favorites:', error)
      return { data: null, error: error.message }
    }

    // Extract recipes from the joined data
    const recipes = data?.map((fav: any) => ({
      ...fav.recipes,
      favorite_id: fav.id,
      favorited_at: fav.created_at
    })) || []

    return { data: recipes, error: null }
  } catch (error) {
    console.error('Error fetching favorites:', error)
    return { data: null, error: error instanceof Error ? error.message : 'Unknown error' }
  }
}

/**
 * Get favorite recipe IDs for a user (for quick lookup)
 */
export async function getUserFavoriteIds(userId: string): Promise<{ data: number[] | null; error: string | null }> {
  try {
    const { data, error } = await supabase
      .from('favorites')
      .select('recipe_id')
      .eq('user_id', userId)

    if (error) {
      console.error('Error fetching favorite IDs:', error)
      return { data: null, error: error.message }
    }

    const recipeIds = data?.map((fav) => fav.recipe_id) || []
    return { data: recipeIds, error: null }
  } catch (error) {
    console.error('Error fetching favorite IDs:', error)
    return { data: null, error: error instanceof Error ? error.message : 'Unknown error' }
  }
}

/**
 * Add a recipe to favorites
 */
export async function addFavorite(userId: string, recipeId: number): Promise<{ success: boolean; error: string | null }> {
  try {
    const { error } = await supabase
      .from('favorites')
      .insert({
        user_id: userId,
        recipe_id: recipeId
      })

    if (error) {
      // Handle duplicate error gracefully
      if (error.code === '23505') {
        return { success: true, error: null } // Already favorited
      }
      console.error('Error adding favorite:', error)
      return { success: false, error: error.message }
    }

    return { success: true, error: null }
  } catch (error) {
    console.error('Error adding favorite:', error)
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' }
  }
}

/**
 * Remove a recipe from favorites
 */
export async function removeFavorite(userId: string, recipeId: number): Promise<{ success: boolean; error: string | null }> {
  try {
    const { error } = await supabase
      .from('favorites')
      .delete()
      .eq('user_id', userId)
      .eq('recipe_id', recipeId)

    if (error) {
      console.error('Error removing favorite:', error)
      return { success: false, error: error.message }
    }

    return { success: true, error: null }
  } catch (error) {
    console.error('Error removing favorite:', error)
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' }
  }
}

/**
 * Toggle favorite status for a recipe
 */
export async function toggleFavorite(
  userId: string,
  recipeId: number,
  isFavorite: boolean
): Promise<{ success: boolean; error: string | null }> {
  if (isFavorite) {
    return removeFavorite(userId, recipeId)
  } else {
    return addFavorite(userId, recipeId)
  }
}

/**
 * Check if a recipe is favorited by a user
 */
export async function isFavorite(userId: string, recipeId: number): Promise<boolean> {
  try {
    const { data, error } = await supabase
      .from('favorites')
      .select('id')
      .eq('user_id', userId)
      .eq('recipe_id', recipeId)
      .single()

    if (error && error.code !== 'PGRST116') { // PGRST116 = no rows found
      console.error('Error checking favorite:', error)
      return false
    }

    return !!data
  } catch (error) {
    console.error('Error checking favorite:', error)
    return false
  }
}
