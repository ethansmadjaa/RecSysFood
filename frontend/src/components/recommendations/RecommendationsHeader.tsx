import { Button } from '@/components/ui/button'
import { ChefHat, RefreshCw, Plus } from 'lucide-react'

type Phase = 'grading' | 'generating_recsys' | 'recsys'

interface RecommendationsHeaderProps {
  phase: Phase
  recommendationsCount: number
  onRefresh: () => void
  onRequestMoreRecipes?: () => void
  refreshing: boolean
  loading: boolean
  requestingMore?: boolean
}

export function RecommendationsHeader({
  phase,
  recommendationsCount,
  onRefresh,
  onRequestMoreRecipes,
  refreshing,
  loading,
  requestingMore,
}: RecommendationsHeaderProps) {
  const getTitle = () => {
    if (phase === 'grading') {
      return 'Note ces recettes'
    }
    if (phase === 'generating_recsys') {
      return 'Génération en cours...'
    }
    return 'Vos recommandations personnalisées'
  }

  const getSubtitle = () => {
    if (phase === 'grading') {
      return 'Ces recettes ont été filtrées à partir de vos préférences alimentaires. Donner votre avis dessus nous permettra de vous recommander des recettes adaptées à vos goûts.'
    }
    if (phase === 'generating_recsys') {
      return 'Nous analysons vos notes pour créer des recommandations sur mesure...'
    }
    if (recommendationsCount > 0) {
      return `${recommendationsCount} recettes sélectionnées spécialement pour vous`
    }
    return 'Basées sur vos préférences et vos goûts'
  }

  return (
    <div className="flex items-center justify-between">
      <div className="flex-1">
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <ChefHat className="h-6 w-6 text-primary" />
          {getTitle()}
        </h1>
        <p className="text-sm text-muted-foreground mt-1 max-w-2xl">{getSubtitle()}</p>
      </div>
      {phase === 'recsys' && (
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={onRequestMoreRecipes}
            disabled={refreshing || loading || requestingMore}
          >
            <Plus className={`h-4 w-4 mr-2 ${requestingMore ? 'animate-pulse' : ''}`} />
            Noter 20 recettes de plus
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={onRefresh}
            disabled={refreshing || loading || requestingMore}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            Rafraîchir
          </Button>
        </div>
      )}
    </div>
  )
}
