import env from '@/constants'
import type { AuthError } from '@supabase/supabase-js'

export interface SignUpData {
  email: string
  password: string
  firstName: string
  lastName: string
}

export interface SignInData {
  email: string
  password: string
}

export interface UserProfile {
  id: string
  auth_id: string
  email: string
  first_name: string | null
  last_name: string | null
  has_completed_signup: boolean
  created_at: string
  updated_at: string
}

interface ApiAuthResponse {
  user: any
  session: any
  error: string | null
}

/**
 * Sign up a new user with email and password via backend API
 */
export async function signUp({ email, password, firstName, lastName }: SignUpData) {
  try {
    const response = await fetch(`${env.API_URL}/auth/signup`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email,
        password,
        firstName,
        lastName,
      }),
    })

    if (!response.ok) {
      const error = await response.json()
      return { user: null, error: { message: error.detail } as AuthError }
    }

    const data: ApiAuthResponse = await response.json()
    return { user: data.user, error: null }
  } catch (error) {
    return { user: null, error: error as AuthError }
  }
}

/**
 * Sign in an existing user with email and password via backend API
 */
export async function signIn({ email, password }: SignInData) {
  try {
    console.log('[API signIn] Starting sign in for:', email)

    const response = await fetch(`${env.API_URL}/auth/signin`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include', // Important for session cookies
      body: JSON.stringify({
        email,
        password,
      }),
    })

    console.log('[API signIn] Response status:', response.status)

    if (!response.ok) {
      const error = await response.json()
      console.log('[API signIn] Error response:', error)
      return { user: null, session: null, error: { message: error.detail } as AuthError }
    }

    const data: ApiAuthResponse = await response.json()
    console.log('[API signIn] Success response:', data)

    return { user: data.user, session: data.session, error: null }
  } catch (error) {
    console.error('[API signIn] Exception:', error)
    return { user: null, session: null, error: error as AuthError }
  }
}

/**
 * Sign out the current user via backend API
 */
export async function signOut() {
  try {
    const response = await fetch(`${env.API_URL}/auth/signout`, {
      method: 'POST',
      credentials: 'include',
    })

    if (!response.ok) {
      const error = await response.json()
      return { error: { message: error.detail } as AuthError }
    }

    return { error: null }
  } catch (error) {
    return { error: error as AuthError }
  }
}

/**
 * Get the current user's profile from the users table via backend API
 */
export async function getUserProfile(authId: string): Promise<UserProfile | null> {
  try {
    console.log('[API getUserProfile] Fetching profile for auth_id:', authId)

    const response = await fetch(`${env.API_URL}/auth/profile/${authId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
    })

    console.log('[API getUserProfile] Response status:', response.status)

    if (!response.ok) {
      const error = await response.json()
      console.error('[API getUserProfile] Error:', error)
      return null
    }

    const data = await response.json()
    console.log('[API getUserProfile] Profile data:', data)
    return data as UserProfile
  } catch (error) {
    console.error('[API getUserProfile] Exception:', error)
    return null
  }
}

/**
 * Update the user's signup completion status via backend API
 */
export async function completeSignup(authId: string) {
  try {
    const response = await fetch(`${env.API_URL}/auth/profile/${authId}/complete-signup`, {
      method: 'PATCH',
      credentials: 'include',
    })

    if (!response.ok) {
      const error = await response.json()
      return { error: new Error(error.detail) }
    }

    return { error: null }
  } catch (error) {
    return { error: error as Error }
  }
}
