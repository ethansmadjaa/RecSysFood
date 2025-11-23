import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import type { SignupFormData } from '@/pages/NewSignupFlow'

interface StepProps {
  data: SignupFormData
  onUpdate: (data: Partial<SignupFormData>) => void
}

const restrictionOptions = [
  { value: 'vegetarian', label: 'Végétarien', description: 'Pas de viande ni de poisson' },
  { value: 'vegan', label: 'Végétalien', description: 'Aucun produit animal' },
  { value: 'no_pork', label: 'Pas de porc', description: 'Sans viande de porc' },
  { value: 'no_alcohol', label: 'Pas d\'alcool', description: 'Sans alcool dans les recettes' },
  { value: 'gluten_free', label: 'Sans gluten (si tu veux)', description: 'Pour les intolérants au gluten' },
  { value: 'none', label: 'Aucune en particulier', description: 'Je mange de tout' }
]

export function StepDietaryRestrictions({ data, onUpdate }: StepProps) {
  const handleToggle = (value: string) => {
    let newRestrictions: string[]

    if (value === 'none') {
      // If "none" is selected, clear all other restrictions
      newRestrictions = data.dietaryRestrictions.includes('none') ? [] : ['none']
    } else {
      // Remove "none" if any other restriction is selected
      const currentRestrictions = data.dietaryRestrictions.filter((r) => r !== 'none')

      newRestrictions = currentRestrictions.includes(value)
        ? currentRestrictions.filter((r) => r !== value)
        : [...currentRestrictions, value]
    }

    onUpdate({ dietaryRestrictions: newRestrictions })
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-foreground mb-2">
          As-tu des restrictions alimentaires ?
        </h2>
        <p className="text-sm text-muted-foreground">
          Plusieurs choix possibles
        </p>
      </div>

      <div className="space-y-3">
        {restrictionOptions.map((option) => (
          <div
            key={option.value}
            className="flex items-start space-x-3 p-4 border rounded-lg hover:bg-accent/50 transition-colors cursor-pointer"
            onClick={() => handleToggle(option.value)}
          >
            <Checkbox
              id={option.value}
              checked={data.dietaryRestrictions.includes(option.value)}
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

      {data.dietaryRestrictions.length > 0 && !data.dietaryRestrictions.includes('none') && (
        <div className="text-center text-sm text-muted-foreground">
          {data.dietaryRestrictions.length} restriction{data.dietaryRestrictions.length > 1 ? 's' : ''} sélectionnée{data.dietaryRestrictions.length > 1 ? 's' : ''}
        </div>
      )}
    </div>
  )
}
