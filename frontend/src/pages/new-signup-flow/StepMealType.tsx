import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import type { SignupFormData } from '@/pages/NewSignupFlow'
import type { MealType } from '@/lib/api/preferences'

interface StepProps {
  data: SignupFormData
  onUpdate: (data: Partial<SignupFormData>) => void
}

const mealTypeOptions: { value: MealType; label: string; description: string }[] = [
  { value: 'breakfast_brunch', label: 'Petit-déjeuner / brunch', description: 'Pour bien démarrer la journée' },
  { value: 'main_course', label: 'Plat principal', description: 'Déjeuner ou dîner' },
  { value: 'starter_side', label: 'Entrée / accompagnement', description: 'Compléter un repas' },
  { value: 'dessert', label: 'Dessert', description: 'Finir en douceur' },
  { value: 'snack', label: 'Snack / goûter', description: 'Petit creux dans la journée' }
]

export function StepMealType({ data, onUpdate }: StepProps) {
  const handleToggle = (value: MealType) => {
    const currentTypes = data.mealTypes
    const newTypes = currentTypes.includes(value)
      ? currentTypes.filter((t) => t !== value)
      : [...currentTypes, value]

    onUpdate({ mealTypes: newTypes })
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-foreground mb-2">
          Tu cherches plutôt quel type de plat ?
        </h2>
        <p className="text-sm text-muted-foreground">
          Plusieurs choix possibles
        </p>
      </div>

      <div className="space-y-3">
        {mealTypeOptions.map((option) => (
          <div
            key={option.value}
            className="flex items-start space-x-3 p-4 border rounded-lg hover:bg-accent/50 transition-colors cursor-pointer"
            onClick={() => handleToggle(option.value)}
          >
            <Checkbox
              id={option.value}
              checked={data.mealTypes.includes(option.value)}
              onCheckedChange={() => handleToggle(option.value)}
              className="mt-1"
            />
            <div className="flex-1">
              <Label
                htmlFor={option.value}
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
      </div>

      {data.mealTypes.length > 0 && (
        <div className="text-center text-sm text-muted-foreground">
          {data.mealTypes.length} type{data.mealTypes.length > 1 ? 's' : ''} sélectionné{data.mealTypes.length > 1 ? 's' : ''}
        </div>
      )}
    </div>
  )
}
