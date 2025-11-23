import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Label } from '@/components/ui/label'
import type { SignupFormData } from '@/pages/NewSignupFlow'

interface StepProps {
  data: SignupFormData
  onUpdate: (data: Partial<SignupFormData>) => void
}

const timeOptions = [
  { value: 15, label: 'Moins de 15 min', description: 'Rapide et efficace' },
  { value: 30, label: 'Moins de 30 min', description: 'Un bon équilibre' },
  { value: 45, label: 'Moins de 45 min', description: 'Je prends mon temps' },
  { value: 60, label: "Jusqu'à 1h", description: 'Pour les occasions spéciales' },
  { value: null, label: 'Pas important', description: 'Le temps ne compte pas' }
]

export function StepCookingTime({ data, onUpdate }: StepProps) {
  const handleChange = (value: string) => {
    const timeValue = value === 'null' ? null : parseInt(value)
    onUpdate({ maxTotalTime: timeValue })
  }

  const currentValue = data.maxTotalTime === null ? 'null' : String(data.maxTotalTime)

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-foreground mb-2">
          Combien de temps tu as pour cuisiner ?
        </h2>
        <p className="text-sm text-muted-foreground">
          On s'adapte à ton emploi du temps
        </p>
      </div>

      <RadioGroup value={currentValue} onValueChange={handleChange} className="space-y-3">
        {timeOptions.map((option) => (
          <div
            key={String(option.value)}
            className="flex items-start space-x-3 p-4 border rounded-lg hover:bg-accent/50 transition-colors cursor-pointer"
            onClick={() => handleChange(String(option.value))}
          >
            <RadioGroupItem
              value={String(option.value)}
              id={`time-${option.value}`}
              className="mt-1"
            />
            <div className="flex-1">
              <Label
                htmlFor={`time-${option.value}`}
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
