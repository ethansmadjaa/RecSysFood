import { useEffect, useState } from 'react'
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
import {
  RecipeCard,
  RecipeDetailDialog,
  GradingSection,
  GradingComplete,
  RecipeCardSkeleton,
  RecommendationsHeader,
  EmptyState,
  GeneratingState,
} from '@/components/recommendations'
import {
  getUserRecommendations,
  triggerRecommendations,
  type Recipe,
  type RecommendationsResponse,
} from '@/lib/api/recommendations'
import { getUserProfile } from '@/lib/api/auth'
import { getUserFavoriteIds, toggleFavorite } from '@/lib/api/favorites'
import { createInteraction, completeGrading, getUserInteractions } from '@/lib/api/interactions'
import { toast } from 'sonner'

export function Recommendations() {
  const { user } = useAuth()
  const [recommendations, setRecommendations] = useState<Recipe[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [status, setStatus] = useState<RecommendationsResponse['status']>('generating')
  const [selectedRecipe, setSelectedRecipe] = useState<Recipe | null>(null)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [favoriteIds, setFavoriteIds] = useState<Set<number>>(new Set())
  const [userProfileId, setUserProfileId] = useState<string | null>(null)

  // Grading state
  const [currentRecipeIndex, setCurrentRecipeIndex] = useState(0)
  const [isGrading, setIsGrading] = useState(true)
  const [gradingComplete, setGradingComplete] = useState(false)
  const [submittingRating, setSubmittingRating] = useState(false)
  const [swipeDirection, setSwipeDirection] = useState<'left' | 'right' | 'down' | null>(null)
  const [recipesToGrade, setRecipesToGrade] = useState<Recipe[]>([])

  const fetchFavorites = async (profileId: string) => {
    const { data } = await getUserFavoriteIds(profileId)
    if (data) {
      setFavoriteIds(new Set(data))
    }
  }

  const fetchRecommendations = async () => {
    if (!user) return

    try {
      const userProfile = await getUserProfile(user.id)
      if (!userProfile) return

      setUserProfileId(userProfile.id)

      // Fetch favorites and interactions in parallel
      fetchFavorites(userProfile.id)
      const { data: interactions } = await getUserInteractions(userProfile.id)

      // Get already graded recipe IDs
      const gradedIds = new Set<number>(interactions?.map((i) => i.recipe_id) || [])

      const { data } = await getUserRecommendations(userProfile.id)

      console.log('data', data)

      if (data) {
        setStatus(data.status)
        if (data.status === 'ready' && data.recipes.length > 0) {
          setRecommendations(data.recipes)

          // Filter out already graded recipes
          const ungraded = data.recipes.filter((r) => !gradedIds.has(r.recipeid))
          setRecipesToGrade(ungraded)

          // If all recipes are already graded, skip grading mode
          if (ungraded.length === 0) {
            setIsGrading(false)
            setGradingComplete(true)
          }

          setLoading(false)
          setRefreshing(false)
          return true
        } else if (data.status === 'not_found') {
          setLoading(false)
          setRefreshing(false)
          return true
        }
      }
      return false
    } catch (error) {
      console.error('Error fetching recommendations:', error)
      setLoading(false)
      setRefreshing(false)
      return true
    }
  }

  const handleToggleFavorite = async (recipeId: number) => {
    if (!userProfileId) return

    const isFav = favoriteIds.has(recipeId)

    // Optimistic update
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
      // Revert on error
      setFavoriteIds((prev) => {
        const newSet = new Set(prev)
        if (isFav) {
          newSet.add(recipeId)
        } else {
          newSet.delete(recipeId)
        }
        return newSet
      })
      toast.error(error || 'Erreur lors de la mise a jour des favoris')
    } else {
      toast.success(isFav ? 'Retire des favoris' : 'Ajoute aux favoris')
    }
  }

  const handleRating = async (rating: 0 | 1 | 2) => {
    if (!userProfileId || currentRecipeIndex >= recipesToGrade.length || submittingRating)
      return

    const currentRecipe = recipesToGrade[currentRecipeIndex]
    setSubmittingRating(true)

    // Set swipe direction based on rating
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

    // Wait for animation to complete before moving to next
    setTimeout(() => {
      setSwipeDirection(null)
      const nextIndex = currentRecipeIndex + 1
      setCurrentRecipeIndex(nextIndex)
      setSubmittingRating(false)

      // Check if all recipes have been graded
      if (nextIndex >= recipesToGrade.length) {
        setGradingComplete(true)
        setIsGrading(false)

        // Update has_graded in the database
        completeGrading(userProfileId).then(({ error: gradingError }) => {
          if (gradingError) {
            toast.error('Erreur lors de la finalisation')
          } else {
            toast.success('Merci pour tes notes ! Tes recommandations vont s\'ameliorer.')
          }
        })
      }
    }, 300)
  }

  useEffect(() => {
    let intervalId: ReturnType<typeof setInterval> | null = null

    const pollRecommendations = async () => {
      const done = await fetchRecommendations()
      if (done && intervalId) {
        clearInterval(intervalId)
      }
    }

    pollRecommendations()
    intervalId = setInterval(pollRecommendations, 2000)

    const timeoutId = setTimeout(() => {
      if (intervalId) clearInterval(intervalId)
      setLoading(false)
    }, 30000)

    return () => {
      if (intervalId) clearInterval(intervalId)
      clearTimeout(timeoutId)
    }
  }, [user])

  const handleRefresh = async () => {
    if (!user) return

    setRefreshing(true)
    setStatus('generating')

    try {
      const userProfile = await getUserProfile(user.id)
      if (!userProfile) return

      await triggerRecommendations(userProfile.id)

      let attempts = 0
      const maxAttempts = 15
      const pollInterval = setInterval(async () => {
        attempts++
        const done = await fetchRecommendations()
        if (done || attempts >= maxAttempts) {
          clearInterval(pollInterval)
          setRefreshing(false)
        }
      }, 2000)
    } catch (error) {
      console.error('Error refreshing recommendations:', error)
      setRefreshing(false)
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

        <div className="flex flex-1 flex-col gap-6 p-4 md:p-6">
          <RecommendationsHeader
            isGrading={isGrading}
            gradingComplete={gradingComplete}
            recommendationsCount={recommendations.length}
            onRefresh={handleRefresh}
            refreshing={refreshing}
            loading={loading}
          />

          {/* Loading state - generating */}
          {(loading || refreshing) && status === 'generating' && <GeneratingState />}

          {/* Loading state - fetching */}
          {loading && status !== 'generating' && (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {[...Array(8)].map((_, i) => (
                <RecipeCardSkeleton key={i} />
              ))}
            </div>
          )}

          {/* Grading mode - Tinder style */}
          {!loading &&
            !refreshing &&
            recipesToGrade.length > 0 &&
            isGrading &&
            !gradingComplete &&
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

          {/* Grading complete */}
          {!loading && !refreshing && gradingComplete && (
            <GradingComplete onViewRecipes={() => setIsGrading(false)} />
          )}

          {/* Recipes grid - after grading is complete */}
          {!loading && !refreshing && recommendations.length > 0 && !isGrading && (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {recommendations.map((recipe) => (
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

          {/* Empty state */}
          {!loading &&
            !refreshing &&
            recommendations.length === 0 &&
            status === 'not_found' && (
              <EmptyState onGenerateRecommendations={handleRefresh} />
            )}
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
