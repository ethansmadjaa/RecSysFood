import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ChefHat } from 'lucide-react'

interface EmptyStateProps {
  onGenerateRecommendations: () => void
}

export function EmptyState({ onGenerateRecommendations }: EmptyStateProps) {
  return (
    <Card className="border-dashed border-2">
      <CardContent className="flex flex-col items-center justify-center py-16">
        <div className="rounded-full bg-muted p-4">
          <ChefHat className="h-12 w-12 text-muted-foreground" />
        </div>
        <p className="text-lg font-semibold mt-6">Pas encore de recommandations</p>
        <p className="text-sm text-muted-foreground mt-2 text-center max-w-md">
          Complete ton profil et tes preferences alimentaires pour recevoir des
          recommandations personnalisees
        </p>
        <Button className="mt-6" onClick={onGenerateRecommendations}>
          Generer mes recommandations
        </Button>
      </CardContent>
    </Card>
  )
}
