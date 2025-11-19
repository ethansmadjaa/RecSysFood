import { createContext, useEffect, useState, type ReactNode, useCallback } from 'react'
import { supabase } from '@/lib/supabase/client'
import { getUserProfile, type UserProfile } from '@/lib/api/auth'
import type { User, Session } from '@supabase/supabase-js'

interface AuthContextType {
  user: User | null
  profile: UserProfile | null
  session: Session | null
  loading: boolean
  refreshProfile: () => Promise<void>
  setAuthData: (user: User | null, profile: UserProfile | null) => void
}

export const AuthContext = createContext<AuthContextType | undefined>(undefined)

interface AuthProviderProps {
  children: ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null)
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [loading, setLoading] = useState(true)

  const refreshProfile = async () => {
    if (!user) {
      setProfile(null)
      return
    }

    const userProfile = await getUserProfile(user.id)
    setProfile(userProfile)
  }

  // Allow manual setting of auth data (for login flow)
  const setAuthData = useCallback((newUser: User | null, newProfile: UserProfile | null) => {
    console.log('[AuthContext] setAuthData called:', { newUser, newProfile })
    setUser(newUser)
    setProfile(newProfile)
  }, [])

  useEffect(() => {
    // Get initial session from Supabase
    const initializeAuth = async () => {
      try {
        console.log('[AuthContext] Initializing auth...')

        // Try to get session from Supabase (with timeout)
        const timeoutPromise = new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Session fetch timeout')), 5000)
        )

        const sessionResult = await Promise.race([
          supabase.auth.getSession(),
          timeoutPromise
        ]).catch((error) => {
          console.warn('[AuthContext] Could not fetch Supabase session:', error)
          return null
        })

        if (sessionResult && typeof sessionResult === 'object' && 'data' in sessionResult) {
          const initialSession = (sessionResult as any).data.session
          console.log('[AuthContext] Initial session:', initialSession)
          setSession(initialSession)

          if (initialSession?.user) {
            console.log('[AuthContext] Found user, fetching profile...')
            setUser(initialSession.user)
            const userProfile = await getUserProfile(initialSession.user.id)
            console.log('[AuthContext] Profile loaded:', userProfile)
            setProfile(userProfile)
          } else {
            console.log('[AuthContext] No user found')
          }
        } else {
          console.log('[AuthContext] No session available')
        }
      } catch (error) {
        console.error('[AuthContext] Error initializing auth:', error)
      } finally {
        console.log('[AuthContext] Setting loading to false')
        setLoading(false)
      }
    }

    initializeAuth()

    // Listen for auth changes from Supabase
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, newSession) => {
        console.log('[AuthContext] Auth state changed:', event, newSession)
        setSession(newSession)
        setUser(newSession?.user ?? null)

        if (newSession?.user) {
          console.log('[AuthContext] Loading profile for user:', newSession.user.id)
          const userProfile = await getUserProfile(newSession.user.id)
          console.log('[AuthContext] Profile loaded:', userProfile)
          setProfile(userProfile)
        } else {
          console.log('[AuthContext] No user, clearing profile')
          setProfile(null)
        }

        setLoading(false)
      }
    )

    return () => {
      subscription.unsubscribe()
    }
  }, [])

  const value = {
    user,
    profile,
    session,
    loading,
    refreshProfile,
    setAuthData,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
