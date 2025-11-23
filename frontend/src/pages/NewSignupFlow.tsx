import { useState } from 'react'
import { useNavigate } from 'react-router'
import { useAuth } from '@/hooks/useAuth'
import { completeSignup, getUserProfile } from '@/lib/api/auth'
import { createUserPreferences, type MealType, type CalorieGoal, type ProteinGoal } from '@/lib/api/preferences'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Card, CardFooter, CardHeader } from '@/components/ui/card'

// Import new step components
import { StepMealType } from '@/pages/new-signup-flow/StepMealType'
import { StepCookingTime } from '@/pages/new-signup-flow/StepCookingTime'
import { StepNutritionGoal } from '@/pages/new-signup-flow/StepNutritionGoal'
import { StepDietaryRestrictions } from '@/pages/new-signup-flow/StepDietaryRestrictions'
import { StepConfirmation } from '@/pages/new-signup-flow/StepConfirmation'

const TOTAL_STEPS = 5

export interface SignupFormData {
  mealTypes: MealType[]
  maxTotalTime: number | null
  calorieGoal: CalorieGoal
  proteinGoal: ProteinGoal
  dietaryRestrictions: string[]
}

export function NewSignupFlow() {
  const navigate = useNavigate()
  const { user, refreshProfile } = useAuth()
  const [currentStep, setCurrentStep] = useState(1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [formData, setFormData] = useState<SignupFormData>({
    mealTypes: [],
    maxTotalTime: null,
    calorieGoal: 'medium',
    proteinGoal: 'medium',
    dietaryRestrictions: []
  })

  const progress = (currentStep / TOTAL_STEPS) * 100

  const handleNext = () => {
    // Validation for each step
    if (currentStep === 1 && formData.mealTypes.length === 0) {
      setError('Veuillez sélectionner au moins un type de plat')
      return
    }

    setError(null)
    if (currentStep < TOTAL_STEPS) {
      setCurrentStep((prev) => prev + 1)
    }
  }

  const handleBack = () => {
    setError(null)
    if (currentStep > 1) {
      setCurrentStep((prev) => prev - 1)
    }
  }

  const handleComplete = async () => {
    if (!user) {
      setError('Utilisateur non connecté')
      return
    }

    setLoading(true)
    setError(null)

    const UserObject = await getUserProfile(user.id)
    if (!UserObject) {
      setError('Utilisateur non trouvé')
      return
    }

    try {
      // Save preferences to database
      const { error: prefsError } = await createUserPreferences({
        user_id: UserObject.id,
        meal_types: formData.mealTypes,
        max_total_time: formData.maxTotalTime,
        calorie_goal: formData.calorieGoal,
        protein_goal: formData.proteinGoal,
        dietary_restrictions: formData.dietaryRestrictions
      })

      if (prefsError) {
        setError(prefsError as string)
        return
      }

      // Mark signup as complete
      const { error: signupError } = await completeSignup(UserObject.auth_id)

      if (signupError) {
        setError(signupError.message)
        return
      }

      // Refresh the user profile
      await refreshProfile()

      // Redirect to dashboard
      navigate('/dashboard')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Une erreur est survenue')
    } finally {
      setLoading(false)
    }
  }

  const updateFormData = (stepData: Partial<SignupFormData>) => {
    setFormData((prev) => ({ ...prev, ...stepData }))
  }

  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return <StepMealType data={formData} onUpdate={updateFormData} />
      case 2:
        return <StepCookingTime data={formData} onUpdate={updateFormData} />
      case 3:
        return <StepNutritionGoal data={formData} onUpdate={updateFormData} />
      case 4:
        return <StepDietaryRestrictions data={formData} onUpdate={updateFormData} />
      case 5:
        return <StepConfirmation data={formData} />
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
              Personnalise ton expérience
            </h1>
            <span className="text-sm text-muted-foreground">
              Question {currentStep} sur {TOTAL_STEPS}
            </span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        <Card>
          <CardHeader>
            <div className="min-h-[400px]">
              {error && (
                <div className="mb-4 p-4 bg-destructive/10 border border-destructive rounded-md">
                  <p className="text-sm text-destructive">{error}</p>
                </div>
              )}
              {renderStep()}
            </div>
          </CardHeader>

          <CardFooter className="flex justify-between">
            <Button
              variant="outline"
              onClick={handleBack}
              disabled={currentStep === 1 || loading}
            >
              Retour
            </Button>

            {currentStep === TOTAL_STEPS ? (
              <Button onClick={handleComplete} disabled={loading}>
                {loading ? 'Finalisation...' : 'Terminer'}
              </Button>
            ) : (
              <Button onClick={handleNext} disabled={loading}>
                Suivant
              </Button>
            )}
          </CardFooter>
        </Card>
      </div>
    </div>
  )
}
