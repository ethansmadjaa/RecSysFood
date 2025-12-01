import env from '@/constants'

export interface InteractionRequest {
  user_id: string
  recipe_id: number
  rating: number // 0 = dislike, 1 = indifferent, 2 = like
}

export interface InteractionResponse {
  interaction_id: number
  user_id: string
  recipe_id: number
  rating: number
  created_at: string
}

/**
 * Create a new interaction (rating) for a recipe
 */
export async function createInteraction(interaction: InteractionRequest): Promise<{ data: InteractionResponse | null; error: string | null }> {
  try {
    const response = await fetch(`${env.API_URL}/api/interactions/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: JSON.stringify(interaction),
    })

    if (!response.ok) {
      const error = await response.json()
      return { data: null, error: error.detail || 'Failed to create interaction' }
    }

    const data: InteractionResponse = await response.json()
    return { data, error: null }
  } catch (error) {
    return { data: null, error: error instanceof Error ? error.message : 'Unknown error' }
  }
}

/**
 * Get all interactions for a user
 */
export async function getUserInteractions(userId: string): Promise<{ data: InteractionResponse[] | null; error: string | null }> {
  try {
    const response = await fetch(`${env.API_URL}/api/interactions/${userId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
    })

    if (!response.ok) {
      if (response.status === 404) {
        return { data: [], error: null }
      }
      const error = await response.json()
      return { data: null, error: error.detail || 'Failed to get interactions' }
    }

    const data: InteractionResponse[] = await response.json()
    return { data, error: null }
  } catch (error) {
    return { data: null, error: error instanceof Error ? error.message : 'Unknown error' }
  }
}

/**
 * Mark user as having completed grading all recommendations
 */
export async function completeGrading(userId: string): Promise<{ success: boolean; error: string | null }> {
  try {
    const response = await fetch(`${env.API_URL}/api/interactions/${userId}/complete-grading`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
    })

    if (!response.ok) {
      const error = await response.json()
      return { success: false, error: error.detail || 'Failed to complete grading' }
    }

    return { success: true, error: null }
  } catch (error) {
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' }
  }
}
