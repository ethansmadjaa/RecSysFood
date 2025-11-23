import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Label } from '@/components/ui/label'
import type { SignupFormData } from '@/pages/NewSignupFlow'
import type { CalorieGoal, ProteinGoal } from '@/lib/api/preferences'

interface StepProps {
  data: SignupFormData
  onUpdate: (data: Partial<SignupFormData>) => void
}

type NutritionChoice = 'light' | 'protein' | 'no_preference'

const nutritionOptions: { value: NutritionChoice; label: string; description: string }[] = [
  { value: 'light', label: 'Plutôt léger (calories modérées)', description: 'Pour garder la ligne' },
  { value: 'protein', label: 'Plutôt protéiné', description: 'Pour la performance et la satiété' },
  { value: 'no_preference', label: 'Je veux juste que ce soit bon, peu importe', description: 'Le goût avant tout' }
]

export function StepNutritionGoal({ data, onUpdate }: StepProps) {
  const getCurrentChoice = (): NutritionChoice => {
    if (data.calorieGoal === 'low') return 'light'
    if (data.proteinGoal === 'high') return 'protein'
    return 'no_preference'
  }

  const handleChange = (value: NutritionChoice) => {
    let calorieGoal: CalorieGoal = 'medium'
    let proteinGoal: ProteinGoal = 'medium'

    switch (value) {
      case 'light':
        calorieGoal = 'low'
        proteinGoal = 'medium'
        break
      case 'protein':
        calorieGoal = 'medium'
        proteinGoal = 'high'
        break
      case 'no_preference':
        calorieGoal = 'medium'
        proteinGoal = 'medium'
        break
    }

    onUpdate({ calorieGoal, proteinGoal })
  }

  const currentValue = getCurrentChoice()

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-foreground mb-2">
          Tu veux quel type de plat ?
        </h2>
        <p className="text-sm text-muted-foreground">
          Choisis selon tes objectifs
        </p>
      </div>

      <RadioGroup value={currentValue} onValueChange={handleChange} className="space-y-3">
        {nutritionOptions.map((option) => (
          <div
            key={option.value}
            className="flex items-start space-x-3 p-4 border rounded-lg hover:bg-accent/50 transition-colors cursor-pointer"
            onClick={() => handleChange(option.value)}
          >
            <RadioGroupItem
              value={option.value}
              id={`nutrition-${option.value}`}
              className="mt-1"
            />
            <div className="flex-1">
              <Label
                htmlFor={`nutrition-${option.value}`}
                className="text-base font-medium cursor-pointer"
              >
                {option.label}
              </Label>
              <p className="text-sm text-muted-foreground mt-1">
                {option.description}
              </p>
            </div>
          </div>
        ))}
      </RadioGroup>
    </div>
  )
}
