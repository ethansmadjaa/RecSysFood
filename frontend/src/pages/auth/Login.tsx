import { useState } from 'react'
import { useNavigate, Link } from 'react-router'
import { Eye, EyeOff } from 'lucide-react'
import { AuthLayout } from '@/components/layouts/AuthLayout'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Card, CardContent, CardFooter } from '@/components/ui/card'
import { getUserProfile } from '@/lib/api/auth'
import { signIn } from '@/lib/supabase/auth'

export function Login() {
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)

    try {
      // Use Supabase sign-in directly so the AuthContext picks up the session
      console.log('[Login] Starting Supabase sign-in...')
      const { user, error: signInError } = await signIn({ email, password })
      console.log('[Login] Sign in result:', { user, signInError })

      if (signInError) {
        setError(signInError.message)
        return
      }

      if (user) {
        console.log('[Login] User logged in successfully!')
        console.log('[Login] Supabase will trigger onAuthStateChange, waiting for redirect...')

        // The AuthContext will pick up the session change via onAuthStateChange
        // and update the user/profile state. We'll navigate based on profile status.
        const profile = await getUserProfile(user.id)
        console.log('[Login] Profile fetched:', profile)

        // Navigate based on signup completion
        if (profile && !profile.has_completed_signup) {
          console.log('[Login] Navigating to signup-flow')
          navigate('/signup-flow')
        } else {
          console.log('[Login] Navigating to dashboard')
          navigate('/dashboard')
        }
      }
    } catch (err) {
      setError('An unexpected error occurred. Please try again.')
      console.error('[Login] Login error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthLayout
      title="Welcome Back"
      description="Sign in to your RecSysFood account"
    >
      <Card>
        <form onSubmit={handleSubmit}>
          <CardContent className="space-y-4 pt-8">
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                disabled={loading}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  disabled={loading}
                  className="pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  disabled={loading}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>
          </CardContent>

          <CardFooter className="flex flex-col space-y-4 pt-4">
            <Button
              type="submit"
              className="w-full"
              disabled={loading}
            >
              {loading ? 'Signing in...' : 'Sign In'}
            </Button>

            <p className="text-sm text-center text-muted-foreground">
              Don't have an account?{' '}
              <Link to="/auth/signup" className="text-primary hover:underline">
                Sign up
              </Link>
            </p>
          </CardFooter>
        </form>
      </Card>
    </AuthLayout>
  )
}
