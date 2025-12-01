import { Button } from '@/components/ui/button'
import { ChefHat, RefreshCw } from 'lucide-react'

interface RecommendationsHeaderProps {
  isGrading: boolean
  gradingComplete: boolean
  recommendationsCount: number
  onRefresh: () => void
  refreshing: boolean
  loading: boolean
}

export function RecommendationsHeader({
  isGrading,
  gradingComplete,
  recommendationsCount,
  onRefresh,
  refreshing,
  loading,
}: RecommendationsHeaderProps) {
  const getTitle = () => {
    if (isGrading && !gradingComplete) {
      return 'Note ces recettes'
    }
    return 'Recommandations personnalisees'
  }

  const getSubtitle = () => {
    if (isGrading && !gradingComplete) {
      return 'Dis-nous ce que tu aimes pour ameliorer tes recommandations'
    }
    if (recommendationsCount > 0) {
      return `${recommendationsCount} recettes selectionnees selon tes preferences`
    }
    return 'Basees sur tes preferences et objectifs alimentaires'
  }

  return (
    <div className="flex items-center justify-between">
      <div>
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <ChefHat className="h-6 w-6 text-primary" />
          {getTitle()}
        </h1>
        <p className="text-sm text-muted-foreground mt-1">{getSubtitle()}</p>
      </div>
      {!isGrading && (
        <Button
          variant="outline"
          size="sm"
          onClick={onRefresh}
          disabled={refreshing || loading}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
          Rafraichir
        </Button>
      )}
    </div>
  )
}
