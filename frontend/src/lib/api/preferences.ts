import { supabase } from '@/lib/supabase/client'

export type MealType = 'breakfast_brunch' | 'main_course' | 'starter_side' | 'dessert' | 'snack'
export type CalorieGoal = 'low' | 'medium' | 'high'
export type ProteinGoal = 'low' | 'medium' | 'high'

export interface UserPreferences {
  user_id: string
  meal_types: MealType[]
  max_total_time: number | null
  calorie_goal: CalorieGoal
  protein_goal: ProteinGoal
  dietary_restrictions: string[]
  allergy_nuts: boolean
  allergy_dairy: boolean
  allergy_egg: boolean
  allergy_fish: boolean
  allergy_soy: boolean
}

export interface UserPreferencesResponse {
  user_preferences_id: number
  user_id: string
  meal_types: MealType[]
  max_total_time: number | null
  calorie_goal: CalorieGoal
  protein_goal: ProteinGoal
  dietary_restrictions: string[]
  allergy_nuts: boolean
  allergy_dairy: boolean
  allergy_egg: boolean
  allergy_fish: boolean
  allergy_soy: boolean
  created_at: string
  updated_at: string
}

import env from '@/constants'

const API_URL = env.API_URL

export async function createUserPreferences(
  preferences: UserPreferences
): Promise<{ data: UserPreferencesResponse | null; error: string | null }> {
  try {
    const { data: { session } } = await supabase.auth.getSession()

    const response = await fetch(`${API_URL}/api/preferences/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(session?.access_token && { Authorization: `Bearer ${session.access_token}` })
      },
      body: JSON.stringify(preferences)
    })

    if (!response.ok) {
      const errorData = await response.json()
      return { data: null, error: errorData.detail || 'Failed to save preferences' }
    }

    const data = await response.json()
    return { data, error: null }
  } catch (error) {
    return { data: null, error: error instanceof Error ? error.message : 'Unknown error' }
  }
}

export async function getUserPreferences(
  userId: string
): Promise<{ data: UserPreferencesResponse | null; error: string | null }> {
  try {
    const { data: { session } } = await supabase.auth.getSession()

    const response = await fetch(`${API_URL}/api/preferences/${userId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...(session?.access_token && { Authorization: `Bearer ${session.access_token}` })
      }
    })

    if (!response.ok) {
      if (response.status === 404) {
        return { data: null, error: null }
      }
      const errorData = await response.json()
      return { data: null, error: errorData.detail || 'Failed to fetch preferences' }
    }

    const data = await response.json()
    return { data, error: null }
  } catch (error) {
    return { data: null, error: error instanceof Error ? error.message : 'Unknown error' }
  }
}
