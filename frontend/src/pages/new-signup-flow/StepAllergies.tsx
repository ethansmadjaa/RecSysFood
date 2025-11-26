import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import type { SignupFormData } from '@/pages/NewSignupFlow'

interface StepProps {
  data: SignupFormData
  onUpdate: (data: Partial<SignupFormData>) => void
}

type AllergyKey = 'allergyNuts' | 'allergyDairy' | 'allergyEgg' | 'allergyFish' | 'allergySoy'

const allergyOptions: { key: AllergyKey; label: string; description: string }[] = [
  { key: 'allergyNuts', label: 'Fruits a coque', description: 'Noix, amandes, noisettes, etc.' },
  { key: 'allergyDairy', label: 'Produits laitiers', description: 'Lait, fromage, beurre, etc.' },
  { key: 'allergyEgg', label: 'Oeufs', description: 'Oeufs et derives' },
  { key: 'allergyFish', label: 'Poisson', description: 'Poisson et fruits de mer' },
  { key: 'allergySoy', label: 'Soja', description: 'Soja et derives' }
]

export function StepAllergies({ data, onUpdate }: StepProps) {
  const handleToggle = (key: AllergyKey) => {
    onUpdate({ [key]: !data[key] })
  }

  const handleNoAllergies = () => {
    // Reset all allergies to false
    onUpdate({
      allergyNuts: false,
      allergyDairy: false,
      allergyEgg: false,
      allergyFish: false,
      allergySoy: false
    })
  }

  const hasAnyAllergy = data.allergyNuts || data.allergyDairy || data.allergyEgg || data.allergyFish || data.allergySoy
  const selectedCount = [data.allergyNuts, data.allergyDairy, data.allergyEgg, data.allergyFish, data.allergySoy].filter(Boolean).length

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-foreground mb-2">
          As-tu des allergies alimentaires ?
        </h2>
        <p className="text-sm text-muted-foreground">
          On evitera ces ingredients dans tes recommandations
        </p>
      </div>

      <div className="space-y-3">
        {allergyOptions.map((option) => (
          <div
            key={option.key}
            className="flex items-start space-x-3 p-4 border rounded-lg hover:bg-accent/50 transition-colors cursor-pointer"
            onClick={() => handleToggle(option.key)}
          >
            <Checkbox
              id={option.key}
              checked={data[option.key]}
              onCheckedChange={() => handleToggle(option.key)}
              className="mt-1"
            />
            <div className="flex-1">
              <Label
                htmlFor={option.key}
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

        {/* Option "Aucune allergie" */}
        <div
          className="flex items-start space-x-3 p-4 border rounded-lg hover:bg-accent/50 transition-colors cursor-pointer"
          onClick={handleNoAllergies}
        >
          <Checkbox
            id="no-allergies"
            checked={!hasAnyAllergy}
            onCheckedChange={handleNoAllergies}
            className="mt-1"
          />
          <div className="flex-1">
            <Label
              htmlFor="no-allergies"
              className="text-base font-medium cursor-pointer"
            >
              Aucune allergie
            </Label>
            <p className="text-sm text-muted-foreground mt-1">
              Je n'ai pas d'allergie alimentaire particuliere
            </p>
          </div>
        </div>
      </div>

      {selectedCount > 0 && (
        <div className="text-center text-sm text-muted-foreground">
          {selectedCount} allergie{selectedCount > 1 ? 's' : ''} selectionnee{selectedCount > 1 ? 's' : ''}
        </div>
      )}
    </div>
  )
}
