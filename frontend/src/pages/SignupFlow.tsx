import { useState } from 'react'
import { useNavigate } from 'react-router'
import { useAuth } from '@/hooks/useAuth'
import { completeSignup } from '@/lib/api/auth'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Card, CardFooter, CardHeader } from '@/components/ui/card'

// Import step components
import { Step1PersonalDetails } from '@/pages/signup-flow/Step1'
import { Step2DietaryPreferences } from '@/pages/signup-flow/Step2'
import { Step3Allergies } from '@/pages/signup-flow/Step3'
import { Step4CuisinePreferences } from '@/pages/signup-flow/Step4'
import { Step5CookingSkill } from '@/pages/signup-flow/Step5'
import { Step6TimePreferences } from '@/pages/signup-flow/Step6'
import { Step7BudgetPreferences } from '@/pages/signup-flow/Step7'
import { Step8Confirmation } from '@/pages/signup-flow/Step8'

const TOTAL_STEPS = 8

export function SignupFlow() {
  const navigate = useNavigate()
  const { user, refreshProfile } = useAuth()
  const [currentStep, setCurrentStep] = useState(1)
  const [loading, setLoading] = useState(false)
  const [formData, setFormData] = useState({})

  const progress = (currentStep / TOTAL_STEPS) * 100

  const handleNext = () => {
    if (currentStep < TOTAL_STEPS) {
      setCurrentStep((prev) => prev + 1)
    }
  }

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep((prev) => prev - 1)
    }
  }

  const handleComplete = async () => {
    if (!user) return

    setLoading(true)
    try {
      const { error } = await completeSignup(user.id)

      if (error) {
        console.error('Error completing signup:', error)
        return
      }

      // Refresh the user profile to get updated has_completed_signup status
      await refreshProfile()

      // Redirect to dashboard
      navigate('/dashboard')
    } catch (error) {
      console.error('Unexpected error completing signup:', error)
    } finally {
      setLoading(false)
    }
  }

  const updateFormData = (stepData: Record<string, unknown>) => {
    setFormData((prev) => ({ ...prev, ...stepData }))
  }

  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return <Step1PersonalDetails data={formData} onUpdate={updateFormData} />
      case 2:
        return <Step2DietaryPreferences data={formData} onUpdate={updateFormData} />
      case 3:
        return <Step3Allergies data={formData} onUpdate={updateFormData} />
      case 4:
        return <Step4CuisinePreferences data={formData} onUpdate={updateFormData} />
      case 5:
        return <Step5CookingSkill data={formData} onUpdate={updateFormData} />
      case 6:
        return <Step6TimePreferences data={formData} onUpdate={updateFormData} />
      case 7:
        return <Step7BudgetPreferences data={formData} onUpdate={updateFormData} />
      case 8:
        return <Step8Confirmation data={formData} onUpdate={updateFormData} />
      default:
        return null
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <div className="w-full max-w-2xl">
        <div className="mb-8">
          <div className="flex justify-between items-center mb-2">
            <h1 className="text-2xl font-bold text-foreground">
              Complete Your Profile
            </h1>
            <span className="text-sm text-muted-foreground">
              Step {currentStep} of {TOTAL_STEPS}
            </span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        <Card>
          <CardHeader>
            <div className="min-h-[400px]">
              {renderStep()}
            </div>
          </CardHeader>

          <CardFooter className="flex justify-between">
            <Button
              variant="outline"
              onClick={handleBack}
              disabled={currentStep === 1 || loading}
            >
              Back
            </Button>

            {currentStep === TOTAL_STEPS ? (
              <Button onClick={handleComplete} disabled={loading}>
                {loading ? 'Completing...' : 'Complete Setup'}
              </Button>
            ) : (
              <Button onClick={handleNext} disabled={loading}>
                Next
              </Button>
            )}
          </CardFooter>
        </Card>
      </div>
    </div>
  )
}
