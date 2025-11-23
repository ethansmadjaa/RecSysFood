import { BrowserRouter, Routes, Route, Navigate } from 'react-router'
import { AuthProvider } from '@/contexts/AuthContext'
import { ProtectedRoute } from '@/components/ProtectedRoute'

// Pages
import { Login } from '@/pages/auth/Login'
import { Signup } from '@/pages/auth/Signup'
import { VerifyEmail } from '@/pages/auth/VerifyEmail'
import { NewSignupFlow } from '@/pages/NewSignupFlow'
import { Dashboard } from '@/pages/Dashboard'
import { Recommendations } from '@/pages/recommendations'
import { Favorites } from '@/pages/favorites'
import { Profile } from '@/pages/profile'
import { Settings } from '@/pages/settings'

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          {/* Public routes */}
          <Route path="/" element={<Navigate to="/auth/login" replace />} />
          <Route path="/auth/login" element={<Login />} />
          <Route path="/auth/signup" element={<Signup />} />
          <Route path="/auth/verify-email" element={<VerifyEmail />} />

          {/* Protected routes - signup flow (doesn't require completed signup) */}
          <Route
            path="/signup-flow"
            element={
              <ProtectedRoute>
                <NewSignupFlow />
              </ProtectedRoute>
            }
          />

          {/* Protected routes - requires completed signup */}
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute requireCompletedSignup>
                <Dashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/recommendations"
            element={
              <ProtectedRoute requireCompletedSignup>
                <Recommendations />
              </ProtectedRoute>
            }
          />
          <Route
            path="/favorites"
            element={
              <ProtectedRoute requireCompletedSignup>
                <Favorites />
              </ProtectedRoute>
            }
          />
          <Route
            path="/profile"
            element={
              <ProtectedRoute requireCompletedSignup>
                <Profile />
              </ProtectedRoute>
            }
          />
          <Route
            path="/settings"
            element={
              <ProtectedRoute requireCompletedSignup>
                <Settings />
              </ProtectedRoute>
            }
          />

          {/* Catch all - redirect to login */}
          <Route path="*" element={<Navigate to="/auth/login" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}

export default App
