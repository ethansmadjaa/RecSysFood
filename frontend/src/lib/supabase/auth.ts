import { supabase } from './client'
import type { User, AuthError } from '@supabase/supabase-js'

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

/**
 * Sign up a new user with email and password
 * Creates auth.users entry and public.users profile automatically via trigger
 */
export async function signUp({ email, password, firstName, lastName }: SignUpData) {
  try {
    // Create auth user
    const { data: authData, error: authError } = await supabase.auth.signUp({
      email,
      password,
    })

    if (authError) throw authError
    if (!authData.user) throw new Error('No user returned from signup')

    // Update the user profile with first and last name
    // (The trigger already created the row with auth_id and email)
    const { error: profileError } = await supabase
      .from('users')
      .update({
        first_name: firstName,
        last_name: lastName,
      })
      .eq('auth_id', authData.user.id)

    if (profileError) throw profileError

    return { user: authData.user, error: null }
  } catch (error) {
    return { user: null, error: error as AuthError }
  }
}

/**
 * Sign in an existing user with email and password
 */
export async function signIn({ email, password }: SignInData) {
  try {
    console.log('[signIn] Starting sign in for:', email)
    console.log('[signIn] Calling supabase.auth.signInWithPassword...')

    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })

    console.log('[signIn] Response received:', { data, error })

    if (error) {
      console.log('[signIn] Error occurred:', error)
      throw error
    }

    console.log('[signIn] Success, returning user and session')
    return { user: data.user, session: data.session, error: null }
  } catch (error) {
    console.log('[signIn] Caught error:', error)
    return { user: null, session: null, error: error as AuthError }
  }
}

/**
 * Sign out the current user
 */
export async function signOut() {
  try {
    const { error } = await supabase.auth.signOut()
    if (error) throw error
    return { error: null }
  } catch (error) {
    return { error: error as AuthError }
  }
}

/**
 * Get the current authenticated user
 */
export async function getCurrentUser(): Promise<User | null> {
  try {
    const { data: { user }, error } = await supabase.auth.getUser()
    if (error) throw error
    return user
  } catch (error) {
    console.error('Error getting current user:', error)
    return null
  }
}

/**
 * Get the current user's profile from the users table
 */
export async function getUserProfile(authId: string): Promise<UserProfile | null> {
  try {
    const { data, error } = await supabase
      .from('users')
      .select('*')
      .eq('auth_id', authId)
      .single()

    if (error) throw error
    return data as UserProfile
  } catch (error) {
    console.error('Error getting user profile:', error)
    return null
  }
}

/**
 * Update the user's signup completion status
 */
export async function completeSignup(authId: string) {
  try {
    const { error } = await supabase
      .from('users')
      .update({ has_completed_signup: true })
      .eq('auth_id', authId)

    if (error) throw error
    return { error: null }
  } catch (error) {
    return { error: error as Error }
  }
}

/**
 * Update user profile information
 */
export async function updateUserProfile(
  authId: string,
  updates: Partial<Omit<UserProfile, 'id' | 'auth_id' | 'created_at' | 'updated_at'>>
) {
  try {
    const { data, error } = await supabase
      .from('users')
      .update(updates)
      .eq('auth_id', authId)
      .select()
      .single()

    if (error) throw error
    return { data: data as UserProfile, error: null }
  } catch (error) {
    return { data: null, error: error as Error }
  }
}
