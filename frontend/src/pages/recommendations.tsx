import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router'
import { useAuth } from '@/hooks/useAuth'
import { SidebarProvider, SidebarInset, SidebarTrigger } from '@/components/ui/sidebar'
import { AppSidebar } from '@/components/AppSidebar'
import { Separator } from '@/components/ui/separator'
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
} from '@/components/ui/breadcrumb'
import { Dialog } from '@/components/ui/dialog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Filter, Sparkles } from 'lucide-react'
import {
  RecipeCard,
  RecipeDetailDialog,
  GradingSection,
  RecipeCardSkeleton,
  EmptyState,
  GeneratingState,
} from '@/components/recommendations'
import {
  getUserRecommendations,
  triggerRecommendations,
  requestMoreRecipesToGrade,
  regenerateRecsysRecommendations,
  type Recipe,
  type RecommendationsResponse,
} from '@/lib/api/recommendations'
import { getUserProfile } from '@/lib/api/auth'
import { getUserFavoriteIds, toggleFavorite } from '@/lib/api/favorites'
import { createInteraction, completeGrading, getUserInteractions } from '@/lib/api/interactions'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { RefreshCw, Plus } from 'lucide-react'

type FilterPhase = 'loading' | 'grading' | 'done'
type RecsysPhase = 'loading' | 'generating' | 'ready'

export function Recommendations() {
  const { user } = useAuth()
  const [searchParams] = useSearchParams()
  const [selectedRecipe, setSelectedRecipe] = useState<Recipe | null>(null)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [favoriteIds, setFavoriteIds] = useState<Set<number>>(new Set())
  const [userProfileId, setUserProfileId] = useState<string | null>(null)
  const initialTab = searchParams.get('tab') || 'filter'
  const [activeTab, setActiveTab] = useState<string>(initialTab)

  // Filter tab state
  const [filterPhase, setFilterPhase] = useState<FilterPhase>('loading')
  const [filterStatus, setFilterStatus] = useState<RecommendationsResponse['status']>('generating')
  const [recipesToGrade, setRecipesToGrade] = useState<Recipe[]>([])
  const [currentRecipeIndex, setCurrentRecipeIndex] = useState(0)
  const [submittingRating, setSubmittingRating] = useState(false)
  const [swipeDirection, setSwipeDirection] = useState<'left' | 'right' | 'down' | null>(null)
  const [filterRefreshing, setFilterRefreshing] = useState(false)
  const [requestingMore, setRequestingMore] = useState(false)

  // Recsys tab state
  const [recsysPhase, setRecsysPhase] = useState<RecsysPhase>('loading')
  const [recsysRecommendations, setRecsysRecommendations] = useState<Recipe[]>([])
  const [recsysRefreshing, setRecsysRefreshing] = useState(false)

  const fetchFavorites = async (profileId: string) => {
    const { data } = await getUserFavoriteIds(profileId)
    if (data) {
      setFavoriteIds(new Set(data))
    }
  }

  // Fetch filter recommendations
  const fetchFilterRecommendations = async () => {
    if (!user) return false

    try {
      const userProfile = await getUserProfile(user.id)
      if (!userProfile) return false

      setUserProfileId(userProfile.id)
      fetchFavorites(userProfile.id)

      const { data: interactions } = await getUserInteractions(userProfile.id)
      const gradedIds = new Set<number>(interactions?.map((i) => i.recipe_id) || [])

      const { data } = await getUserRecommendations(userProfile.id, 'filter')

      if (data) {
        setFilterStatus(data.status)
        if (data.status === 'ready' && data.recipes.length > 0) {
          const ungraded = data.recipes.filter((r) => !gradedIds.has(r.recipeid))
          setRecipesToGrade(ungraded)

          if (ungraded.length === 0) {
            setFilterPhase('done')
          } else {
            setFilterPhase('grading')
          }
          return true
        } else if (data.status === 'not_found') {
          setFilterPhase('done')
          return true
        }
      }
      return false
    } catch (error) {
      console.error('Error fetching filter recommendations:', error)
      setFilterPhase('done')
      return true
    }
  }

  // Fetch recsys recommendations
  const fetchRecsysRecommendations = async () => {
    if (!user) return false

    try {
      const userProfile = await getUserProfile(user.id)
      if (!userProfile) return false

      setUserProfileId(userProfile.id)

      const { data } = await getUserRecommendations(userProfile.id, 'recsys')

      if (data) {
        if (data.status === 'ready' && data.recipes.length > 0) {
          setRecsysRecommendations(data.recipes)
          setRecsysPhase('ready')
          return true
        } else if (data.status === 'generating') {
          setRecsysPhase('generating')
          return false
        } else if (data.status === 'not_found') {
          setRecsysPhase('ready')
          return true
        }
      }
      return false
    } catch (error) {
      console.error('Error fetching recsys recommendations:', error)
      setRecsysPhase('ready')
      return true
    }
  }

  const handleToggleFavorite = async (recipeId: number) => {
    if (!userProfileId) return

    const isFav = favoriteIds.has(recipeId)

    setFavoriteIds((prev) => {
      const newSet = new Set(prev)
      if (isFav) {
        newSet.delete(recipeId)
      } else {
        newSet.add(recipeId)
      }
      return newSet
    })

    const { success, error } = await toggleFavorite(userProfileId, recipeId, isFav)

    if (!success) {
      setFavoriteIds((prev) => {
        const newSet = new Set(prev)
        if (isFav) {
          newSet.add(recipeId)
        } else {
          newSet.delete(recipeId)
        }
        return newSet
      })
      toast.error(error || 'Erreur lors de la mise à jour des favoris')
    } else {
      toast.success(isFav ? 'Retiré des favoris' : 'Ajouté aux favoris')
    }
  }

  const handleRating = async (rating: 0 | 1 | 2) => {
    if (!userProfileId || currentRecipeIndex >= recipesToGrade.length || submittingRating)
      return

    const currentRecipe = recipesToGrade[currentRecipeIndex]
    setSubmittingRating(true)

    const direction = rating === 2 ? 'right' : rating === 0 ? 'left' : 'down'
    setSwipeDirection(direction)

    const { error } = await createInteraction({
      user_id: userProfileId,
      recipe_id: currentRecipe.recipeid,
      rating,
    })

    if (error) {
      toast.error("Erreur lors de l'envoi de la note")
      setSubmittingRating(false)
      setSwipeDirection(null)
      return
    }

    setTimeout(() => {
      setSwipeDirection(null)
      const nextIndex = currentRecipeIndex + 1
      setCurrentRecipeIndex(nextIndex)
      setSubmittingRating(false)

      // Check if all recipes have been graded
      if (nextIndex >= recipesToGrade.length) {
        setFilterPhase('done')

        // Complete grading (triggers recsys generation in background)
        completeGrading(userProfileId).then(({ error: gradingError }) => {
          if (gradingError) {
            toast.error('Erreur lors de la finalisation')
          } else {
            toast.success('Merci pour tes notes ! Les recommandations IA seront bientôt disponibles.')
            // Trigger recsys refresh
            setRecsysPhase('generating')
            pollRecsysAfterGrading()
          }
        })
      }
    }, 300)
  }

  const pollRecsysAfterGrading = async () => {
    let attempts = 0
    const maxAttempts = 60 // 60 attempts * 2s = 2 minutes max (model training can take time)

    const poll = async () => {
      const done = await fetchRecsysRecommendations()
      if (done || attempts >= maxAttempts) {
        return true
      }
      return false
    }

    const intervalId = setInterval(async () => {
      attempts++
      const done = await poll()
      if (done) {
        clearInterval(intervalId)
      }
    }, 2000)

    poll()
  }

  // Initial load for filter tab
  useEffect(() => {
    let intervalId: ReturnType<typeof setInterval> | null = null
    let mounted = true

    const pollRecommendations = async () => {
      if (!mounted) return
      const done = await fetchFilterRecommendations()
      if (done && intervalId) {
        clearInterval(intervalId)
        intervalId = null
      }
    }

    pollRecommendations()
    intervalId = setInterval(pollRecommendations, 2000)

    const timeoutId = setTimeout(() => {
      if (intervalId) {
        clearInterval(intervalId)
        intervalId = null
      }
      if (filterPhase === 'loading') {
        setFilterPhase('done')
      }
    }, 30000)

    return () => {
      mounted = false
      if (intervalId) clearInterval(intervalId)
      clearTimeout(timeoutId)
    }
  }, [user])

  // Initial load for recsys tab
  useEffect(() => {
    let intervalId: ReturnType<typeof setInterval> | null = null
    let mounted = true

    const pollRecommendations = async () => {
      if (!mounted) return
      const done = await fetchRecsysRecommendations()
      if (done && intervalId) {
        clearInterval(intervalId)
        intervalId = null
      }
    }

    pollRecommendations()
    intervalId = setInterval(pollRecommendations, 2000)

    const timeoutId = setTimeout(() => {
      if (intervalId) {
        clearInterval(intervalId)
        intervalId = null
      }
      if (recsysPhase === 'loading') {
        setRecsysPhase('ready')
      }
    }, 120000) // 2 minutes timeout to allow for model training

    return () => {
      mounted = false
      if (intervalId) clearInterval(intervalId)
      clearTimeout(timeoutId)
    }
  }, [user])

  const handleRefreshFilter = async () => {
    if (!user) return

    setFilterRefreshing(true)
    setFilterStatus('generating')
    setFilterPhase('loading')

    try {
      const userProfile = await getUserProfile(user.id)
      if (!userProfile) return

      await triggerRecommendations(userProfile.id)

      let attempts = 0
      const maxAttempts = 15
      const pollInterval = setInterval(async () => {
        attempts++
        const done = await fetchFilterRecommendations()
        if (done || attempts >= maxAttempts) {
          clearInterval(pollInterval)
          setFilterRefreshing(false)
        }
      }, 2000)
    } catch (error) {
      console.error('Error refreshing filter recommendations:', error)
      setFilterRefreshing(false)
      setFilterPhase('done')
    }
  }

  const handleRefreshRecsys = async () => {
    if (!user || !userProfileId) return

    setRecsysRefreshing(true)
    setRecsysRecommendations([])
    setRecsysPhase('generating')

    try {
      await regenerateRecsysRecommendations(userProfileId)

      let attempts = 0
      const maxAttempts = 60 // 60 attempts * 2s = 2 minutes max (model training can take time)

      const pollInterval = setInterval(async () => {
        attempts++
        const { data } = await getUserRecommendations(userProfileId, 'recsys')

        if (data && data.status === 'ready' && data.recipes.length > 0) {
          setRecsysRecommendations(data.recipes)
          setRecsysPhase('ready')
          setRecsysRefreshing(false)
          clearInterval(pollInterval)
          toast.success('Recommandations régénérées avec succès !')
        } else if (attempts >= maxAttempts) {
          setRecsysPhase('ready')
          setRecsysRefreshing(false)
          clearInterval(pollInterval)
          toast.error('Timeout lors de la génération des recommandations')
        }
      }, 2000)
    } catch (error) {
      console.error('Error refreshing recsys recommendations:', error)
      setRecsysRefreshing(false)
      setRecsysPhase('ready')
      toast.error('Erreur lors de la régénération des recommandations')
    }
  }

  const handleRequestMoreRecipes = async () => {
    if (!userProfileId) return

    setRequestingMore(true)

    try {
      const { data, error } = await requestMoreRecipesToGrade(userProfileId)

      if (error) {
        toast.error(error)
        setRequestingMore(false)
        return
      }

      if (data && data.count > 0) {
        toast.success(`${data.count} nouvelles recettes à noter !`)

        const { data: newRecipes } = await getUserRecommendations(userProfileId, 'filter')

        if (newRecipes && newRecipes.recipes.length > 0) {
          setRecipesToGrade(newRecipes.recipes)
          setCurrentRecipeIndex(0)
          setFilterPhase('grading')
        }
      } else {
        toast.info('Aucune nouvelle recette disponible')
      }
    } catch (error) {
      console.error('Error requesting more recipes:', error)
      toast.error('Erreur lors de la demande de recettes supplémentaires')
    } finally {
      setRequestingMore(false)
    }
  }

  const openRecipeDetail = (recipe: Recipe) => {
    setSelectedRecipe(recipe)
    setDialogOpen(true)
  }

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-2 h-4" />
          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem>
                <BreadcrumbPage>Recommandations</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </header>

        <div className="flex flex-1 flex-col gap-4 p-4 md:p-6">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full max-w-md grid-cols-2">
              <TabsTrigger value="filter" className="gap-2">
                <Filter className="h-4 w-4" />
                À noter
              </TabsTrigger>
              <TabsTrigger value="recsys" className="gap-2">
                <Sparkles className="h-4 w-4" />
                Pour toi
              </TabsTrigger>
            </TabsList>

            {/* Filter Tab Content */}
            <TabsContent value="filter" className="mt-6">
              {/* Filter Header */}
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold">Recettes à noter</h2>
                  <p className="text-sm text-muted-foreground">
                    {filterPhase === 'grading' && recipesToGrade.length > 0
                      ? `${recipesToGrade.length - currentRecipeIndex} recette(s) restante(s)`
                      : filterPhase === 'done'
                        ? 'Toutes les recettes ont été notées'
                        : 'Chargement...'}
                  </p>
                </div>
                <div className="flex gap-2">
                  {filterPhase === 'done' && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleRequestMoreRecipes}
                      disabled={requestingMore || filterRefreshing}
                    >
                      {requestingMore ? (
                        <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <Plus className="mr-2 h-4 w-4" />
                      )}
                      Plus de recettes
                    </Button>
                  )}
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleRefreshFilter}
                    disabled={filterRefreshing || requestingMore}
                  >
                    <RefreshCw className={`mr-2 h-4 w-4 ${filterRefreshing ? 'animate-spin' : ''}`} />
                    Actualiser
                  </Button>
                </div>
              </div>

              {/* Filter Loading */}
              {(filterPhase === 'loading' || filterRefreshing) && filterStatus === 'generating' && (
                <GeneratingState />
              )}

              {/* Requesting more recipes */}
              {requestingMore && (
                <GeneratingState message="Chargement de nouvelles recettes à noter..." />
              )}

              {/* Filter Loading skeleton */}
              {filterPhase === 'loading' && filterStatus !== 'generating' && (
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                  {[...Array(8)].map((_, i) => (
                    <RecipeCardSkeleton key={i} />
                  ))}
                </div>
              )}

              {/* Grading mode */}
              {!filterRefreshing &&
                !requestingMore &&
                filterPhase === 'grading' &&
                recipesToGrade.length > 0 &&
                currentRecipeIndex < recipesToGrade.length && (
                  <GradingSection
                    recipesToGrade={recipesToGrade}
                    currentRecipeIndex={currentRecipeIndex}
                    onRate={handleRating}
                    onViewDetails={openRecipeDetail}
                    submittingRating={submittingRating}
                    swipeDirection={swipeDirection}
                  />
                )}

              {/* Filter done - show appropriate message */}
              {!filterRefreshing &&
                !requestingMore &&
                filterPhase === 'done' && (
                  filterStatus === 'not_found' ? (
                    <EmptyState
                      onGenerateRecommendations={handleRefreshFilter}
                      message="Aucune recette à noter pour le moment. Clique sur 'Plus de recettes' pour en obtenir."
                    />
                  ) : (
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                      <div className="rounded-full bg-green-100 p-4 mb-4">
                        <Sparkles className="h-8 w-8 text-green-600" />
                      </div>
                      <h3 className="text-lg font-semibold mb-2">Bravo !</h3>
                      <p className="text-muted-foreground max-w-md mb-6">
                        Tu as noté toutes les recettes disponibles. Consulte l'onglet "Pour toi" pour voir tes recommandations personnalisées, ou demande plus de recettes à noter.
                      </p>
                      <div className="flex gap-3">
                        <Button
                          onClick={() => setActiveTab('recsys')}
                          className="gap-2"
                        >
                          <Sparkles className="h-4 w-4" />
                          Voir mes recommandations
                        </Button>
                        <Button
                          variant="outline"
                          onClick={handleRequestMoreRecipes}
                          disabled={requestingMore}
                          className="gap-2"
                        >
                          {requestingMore ? (
                            <RefreshCw className="h-4 w-4 animate-spin" />
                          ) : (
                            <Plus className="h-4 w-4" />
                          )}
                          Continuer à noter
                        </Button>
                      </div>
                    </div>
                  )
                )}
            </TabsContent>

            {/* Recsys Tab Content */}
            <TabsContent value="recsys" className="mt-6">
              {/* Recsys Header */}
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold">Recommandations personnalisées</h2>
                  <p className="text-sm text-muted-foreground">
                    {recsysPhase === 'ready' && recsysRecommendations.length > 0
                      ? `${recsysRecommendations.length} recette(s) recommandée(s)`
                      : recsysPhase === 'generating'
                        ? 'Génération en cours...'
                        : 'Aucune recommandation'}
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRefreshRecsys}
                  disabled={recsysRefreshing}
                >
                  <RefreshCw className={`mr-2 h-4 w-4 ${recsysRefreshing ? 'animate-spin' : ''}`} />
                  Régénérer
                </Button>
              </div>

              {/* Recsys Loading/Generating */}
              {(recsysPhase === 'loading' || recsysPhase === 'generating' || recsysRefreshing) && (
                <GeneratingState message="Génération de vos recommandations personnalisées..." />
              )}

              {/* Recsys Ready - with recommendations */}
              {!recsysRefreshing && recsysPhase === 'ready' && recsysRecommendations.length > 0 && (
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                  {recsysRecommendations.map((recipe) => (
                    <RecipeCard
                      key={recipe.recipeid}
                      recipe={recipe}
                      isFavorite={favoriteIds.has(recipe.recipeid)}
                      onToggleFavorite={handleToggleFavorite}
                      onViewDetails={openRecipeDetail}
                    />
                  ))}
                </div>
              )}

              {/* Recsys Empty */}
              {!recsysRefreshing && recsysPhase === 'ready' && recsysRecommendations.length === 0 && (
                <EmptyState
                  onGenerateRecommendations={handleRefreshRecsys}
                  message="Aucune recommandation personnalisée disponible. Note d'abord quelques recettes dans l'onglet 'À noter' pour obtenir des suggestions."
                />
              )}
            </TabsContent>
          </Tabs>
        </div>
      </SidebarInset>

      {/* Recipe Detail Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        {selectedRecipe && (
          <RecipeDetailDialog
            recipe={selectedRecipe}
            isFavorite={favoriteIds.has(selectedRecipe.recipeid)}
            onToggleFavorite={handleToggleFavorite}
          />
        )}
      </Dialog>
    </SidebarProvider>
  )
}
