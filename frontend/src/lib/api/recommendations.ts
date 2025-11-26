import { supabase } from '@/lib/supabase/client'
import env from '@/constants'

const API_URL = env.API_URL

export interface Recipe {
  recipeid: number
  name: string
  description: string | null
  authorname: string | null
  datepublished: string | null
  images: string[] | null
  recipecategory: string | null
  keywords: string[] | null
  recipeingredientquantities: string[] | null
  recipeingredientparts: string[] | null
  recipeinstructions: string[] | null
  cooktime_min: number | null
  preptime_min: number | null
  totaltime_min: number | null
  recipeservings: number | null
  recipeyield: string | null
  aggregatedrating: number | null
  reviewcount: number | null
  // Nutrition
  calories: number | null
  fatcontent: number | null
  saturatedfatcontent: number | null
  cholesterolcontent: number | null
  sodiumcontent: number | null
  carbohydratecontent: number | null
  fibercontent: number | null
  sugarcontent: number | null
  proteincontent: number | null
  // Dietary flags
  is_vegan: boolean | null
  is_vegetarian: boolean | null
  contains_pork: boolean | null
  contains_alcohol: boolean | null
  contains_gluten: boolean | null
  contains_nuts: boolean | null
  contains_dairy: boolean | null
  contains_egg: boolean | null
  contains_fish: boolean | null
  contains_soy: boolean | null
  // Meal type flags
  is_breakfast_brunch: boolean | null
  is_dessert: boolean | null
  // Categories
  calorie_category: string | null
  protein_category: string | null
  // Recommendation score
  score: number | null
}

export interface RecommendationsResponse {
  status: 'ready' | 'generating' | 'not_found'
  recipes: Recipe[]
}

export async function triggerRecommendations(userId: string): Promise<void> {
  try {
    const { data: { session } } = await supabase.auth.getSession()

    await fetch(`${API_URL}/api/recommendations/${userId}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(session?.access_token && { Authorization: `Bearer ${session.access_token}` })
      }
    })
  } catch (error) {
    console.error('Failed to trigger recommendations:', error)
  }
}

export async function getUserRecommendations(
  userId: string
): Promise<{ data: RecommendationsResponse | null; error: string | null }> {
  try {
    const { data: { session } } = await supabase.auth.getSession()

    const response = await fetch(`${API_URL}/api/recommendations/${userId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...(session?.access_token && { Authorization: `Bearer ${session.access_token}` })
      }
    })

    if (!response.ok) {
      const errorData = await response.json()
      return { data: null, error: errorData.detail || 'Failed to fetch recommendations' }
    }

    const data = await response.json()
    return { data, error: null }
  } catch (error) {
    return { data: null, error: error instanceof Error ? error.message : 'Unknown error' }
  }
}
