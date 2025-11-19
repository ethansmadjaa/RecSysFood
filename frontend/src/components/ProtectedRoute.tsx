import { Navigate } from 'react-router'
import { useAuth } from '@/hooks/useAuth'
import type { ReactNode } from 'react'

interface ProtectedRouteProps {
  children: ReactNode
  requireCompletedSignup?: boolean
}

export function ProtectedRoute({ children, requireCompletedSignup = false }: ProtectedRouteProps) {
  const { user, profile, loading } = useAuth()

  // Show loading state while checking authentication
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-muted-foreground">Loading...</p>
        </div>
      </div>
    )
  }

  // Redirect to login if not authenticated
  if (!user) {
    return <Navigate to="/auth/login" replace />
  }

  // Check if email is verified (user.email_confirmed_at will be null if not verified)
  if (user && !user.email_confirmed_at) {
    return <Navigate to={`/auth/verify-email?email=${encodeURIComponent(user.email || '')}`} replace />
  }

  // If we require completed signup and user hasn't completed it, redirect to signup flow
  if (requireCompletedSignup && profile && !profile.has_completed_signup) {
    return <Navigate to="/signup-flow" replace />
  }

  return <>{children}</>
}
