import { Badge } from '@/components/ui/badge'
import type { SignupFormData } from '@/pages/NewSignupFlow'
import type { CalorieGoal, ProteinGoal } from '@/lib/api/preferences'

interface StepProps {
  data: SignupFormData
}

const mealTypeLabels: Record<string, string> = {
  breakfast_brunch: 'Petit-déjeuner / brunch',
  main_course: 'Plat principal',
  starter_side: 'Entrée / accompagnement',
  dessert: 'Dessert',
  snack: 'Snack / goûter'
}

const restrictionLabels: Record<string, string> = {
  vegetarian: 'Végétarien',
  vegan: 'Végétalien',
  no_pork: 'Pas de porc',
  no_alcohol: 'Pas d\'alcool',
  gluten_free: 'Sans gluten',
  none: 'Aucune'
}

const nutritionLabels: {
  calorie: Record<CalorieGoal, string>
  protein: Record<ProteinGoal, string>
} = {
  calorie: {
    low: 'Calories modérées',
    medium: 'Calories normales',
    high: 'Calories élevées'
  },
  protein: {
    low: 'Peu de protéines',
    medium: 'Protéines normales',
    high: 'Riche en protéines'
  }
}

export function StepConfirmation({ data }: StepProps) {
  const getTimeLabel = () => {
    if (data.maxTotalTime === null) return 'Pas important'
    if (data.maxTotalTime <= 15) return 'Moins de 15 min'
    if (data.maxTotalTime <= 30) return 'Moins de 30 min'
    if (data.maxTotalTime <= 45) return 'Moins de 45 min'
    return 'Jusqu\'à 1h'
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-foreground mb-2">
          Récapitulatif de tes préférences
        </h2>
        <p className="text-sm text-muted-foreground">
          Vérifie que tout est correct avant de continuer
        </p>
      </div>

      <div className="space-y-4">
        {/* Meal Types */}
        <div className="p-4 border rounded-lg bg-accent/20">
          <h3 className="font-semibold text-sm text-muted-foreground mb-2">
            Types de plats recherchés
          </h3>
          <div className="flex flex-wrap gap-2">
            {data.mealTypes.map((type) => (
              <Badge key={type} variant="secondary">
                {mealTypeLabels[type] || type}
              </Badge>
            ))}
          </div>
        </div>

        {/* Cooking Time */}
        <div className="p-4 border rounded-lg bg-accent/20">
          <h3 className="font-semibold text-sm text-muted-foreground mb-2">
            Temps de cuisine
          </h3>
          <p className="text-base">{getTimeLabel()}</p>
        </div>

        {/* Nutrition Goals */}
        <div className="p-4 border rounded-lg bg-accent/20">
          <h3 className="font-semibold text-sm text-muted-foreground mb-2">
            Objectifs nutritionnels
          </h3>
          <div className="space-y-1">
            <p className="text-base">
              {nutritionLabels.calorie[data.calorieGoal]}
            </p>
            <p className="text-base">
              {nutritionLabels.protein[data.proteinGoal]}
            </p>
          </div>
        </div>

        {/* Dietary Restrictions */}
        <div className="p-4 border rounded-lg bg-accent/20">
          <h3 className="font-semibold text-sm text-muted-foreground mb-2">
            Restrictions alimentaires
          </h3>
          {data.dietaryRestrictions.length === 0 || data.dietaryRestrictions.includes('none') ? (
            <p className="text-base text-muted-foreground">Aucune restriction</p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {data.dietaryRestrictions.map((restriction) => (
                <Badge key={restriction} variant="outline">
                  {restrictionLabels[restriction] || restriction}
                </Badge>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="text-center pt-4">
        <p className="text-sm text-muted-foreground">
          Clique sur "Terminer" pour sauvegarder tes préférences
        </p>
      </div>
    </div>
  )
}
