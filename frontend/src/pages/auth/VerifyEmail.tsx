import { useState, useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router'
import { AuthLayout } from '@/components/layouts/AuthLayout'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Card, CardContent, CardFooter } from '@/components/ui/card'
import { supabase } from '@/lib/supabase/client'

export function VerifyEmail() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const email = searchParams.get('email')

  const [code, setCode] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [resendLoading, setResendLoading] = useState(false)

  useEffect(() => {
    if (!email) {
      navigate('/auth/signup')
    }
  }, [email, navigate])

  const handleVerify = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setSuccess(null)

    if (!email) {
      setError('Email not found. Please sign up again.')
      return
    }

    if (code.length !== 8) {
      setError('Please enter a valid 8-digit code')
      return
    }

    setLoading(true)

    try {
      const { error: verifyError } = await supabase.auth.verifyOtp({
        email,
        token: code,
        type: 'signup'
      })

      if (verifyError) {
        setError(verifyError.message)
        return
      }

      setSuccess('Email verified successfully! Redirecting...')

      // Wait a moment to show success message, then redirect
      setTimeout(() => {
        navigate('/signup-flow')
      }, 1500)
    } catch (err) {
      setError('An unexpected error occurred. Please try again.')
      console.error('Verification error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleResendCode = async () => {
    if (!email) return

    setResendLoading(true)
    setError(null)
    setSuccess(null)

    try {
      const { error: resendError } = await supabase.auth.resend({
        type: 'signup',
        email
      })

      if (resendError) {
        setError(resendError.message)
        return
      }

      setSuccess('Verification code resent! Check your email.')
    } catch (err) {
      setError('Failed to resend code. Please try again.')
      console.error('Resend error:', err)
    } finally {
      setResendLoading(false)
    }
  }

  return (
    <AuthLayout
      title="Verify Your Email"
      description={`We sent a 8-digit code to ${email}`}
    >
      <Card>
        <form onSubmit={handleVerify}>
          <CardContent className="space-y-4 pt-8">
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {success && (
              <Alert>
                <AlertDescription>{success}</AlertDescription>
              </Alert>
            )}

            <div className="space-y-2">
              <Label htmlFor="code">Verification Code</Label>
              <Input
                id="code"
                type="text"
                placeholder="00000000"
                value={code}
                onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 8))}
                maxLength={8}
                required
                disabled={loading}
                className="text-center text-2xl tracking-widest"
              />
              <p className="text-xs text-muted-foreground text-center">
                Enter the 8-digit code from your email
              </p>
            </div>
          </CardContent>

          <CardFooter className="flex flex-col space-y-4">
            <Button
              type="submit"
              className="w-full"
              disabled={loading || code.length !== 8}
            >
              {loading ? 'Verifying...' : 'Verify Email'}
            </Button>

            <div className="text-center space-y-2">
              <p className="text-sm text-muted-foreground">
                Didn't receive the code?
              </p>
              <Button
                type="button"
                variant="outline"
                onClick={handleResendCode}
                disabled={resendLoading}
                className="w-full"
              >
                {resendLoading ? 'Sending...' : 'Resend Code'}
              </Button>
            </div>
          </CardFooter>
        </form>
      </Card>
    </AuthLayout>
  )
}
