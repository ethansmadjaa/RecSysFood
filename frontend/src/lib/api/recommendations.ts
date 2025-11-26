import { supabase } from '@/lib/supabase/client'
import env from '@/constants'

const API_URL = env.API_URL

export interface Recipe {
  recipeid: number
  name: string
  images: string[] | null
  totaltime_min: number | null
  aggregatedrating: number | null
  reviewcount: number | null
  calories: number | null
  proteincontent: number | null
  is_vegan: boolean | null
  is_vegetarian: boolean | null
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
