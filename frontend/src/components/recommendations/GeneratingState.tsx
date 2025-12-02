import { Card, CardContent } from '@/components/ui/card'
import { Loader2, ChefHat } from 'lucide-react'

interface GeneratingStateProps {
  message?: string
}

export function GeneratingState({ message }: GeneratingStateProps) {
  return (
    <Card className="border-dashed border-2">
      <CardContent className="flex flex-col items-center justify-center py-16">
        <div className="relative">
          <Loader2 className="h-16 w-16 text-primary animate-spin" />
          <ChefHat className="h-6 w-6 text-primary absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
        </div>
        <p className="text-lg font-semibold mt-6">
          {message || 'Génération des recommandations...'}
        </p>
        <p className="text-sm text-muted-foreground mt-2 text-center max-w-md">
          On analyse vos préférences pour vous proposer les meilleures recettes. Ça
          ne prendra que quelques secondes !
        </p>
      </CardContent>
    </Card>
  )
}
