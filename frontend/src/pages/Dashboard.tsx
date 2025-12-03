import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router'
import { useAuth } from '@/hooks/useAuth'
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card'
import { SidebarProvider, SidebarInset, SidebarTrigger } from '@/components/ui/sidebar'
import { AppSidebar } from '@/components/AppSidebar'
import { Separator } from '@/components/ui/separator'
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
} from '@/components/ui/breadcrumb'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { AspectRatio } from '@/components/ui/aspect-ratio'
import { Dialog } from '@/components/ui/dialog'
import {
  Heart,
  ChefHat,
  Clock,
  Star,
  Flame,
  ArrowRight,
  Sparkles,
  Filter,
  CheckCircle2,
  AlertCircle,
  ThumbsUp,
  ThumbsDown,
  Meh,
} from 'lucide-react'
import { getUserRecommendations, type Recipe } from '@/lib/api/recommendations'
import { getUserProfile } from '@/lib/api/auth'
import { getUserFavorites, toggleFavorite } from '@/lib/api/favorites'
import { getUserInteractions, createInteraction } from '@/lib/api/interactions'
import { toast } from 'sonner'
import {
  RecipeDetailDialog,
  getRecipeImageUrl,
  DEFAULT_FOOD_IMAGE,
} from '@/components/recommendations'

// Mini carte pour les favoris
function FavoriteCard({
  recipe,
  onViewDetails,
  onToggleFavorite,
}: {
  recipe: Recipe
  onViewDetails: (recipe: Recipe) => void
  onToggleFavorite: (recipeId: number) => void
}) {
  const imageUrl = getRecipeImageUrl(recipe.images, recipe.recipeid)

  return (
    <Card
      className="group overflow-hidden transition-all hover:shadow-md cursor-pointer"
      onClick={() => onViewDetails(recipe)}
    >
      <div className="relative">
        <AspectRatio ratio={16 / 9}>
          <img
            src={imageUrl}
            alt={recipe.name}
            className="h-full w-full object-cover transition-transform group-hover:scale-105"
            onError={(e) => {
              ;(e.target as HTMLImageElement).src = DEFAULT_FOOD_IMAGE
            }}
          />
        </AspectRatio>
        <Button
          variant="secondary"
          size="icon"
          className="absolute right-1.5 top-1.5 h-7 w-7 rounded-full opacity-100"
          onClick={(e) => {
            e.stopPropagation()
            onToggleFavorite(recipe.recipeid)
          }}
        >
          <Heart className="h-3.5 w-3.5 fill-red-500 text-red-500" />
        </Button>
      </div>
      <CardContent className="p-3">
        <h4 className="font-medium text-sm line-clamp-1">{recipe.name}</h4>
        <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
          {recipe.totaltime_min && (
            <span className="flex items-center gap-0.5">
              <Clock className="h-3 w-3" />
              {recipe.totaltime_min}min
            </span>
          )}
          {recipe.aggregatedrating && (
            <span className="flex items-center gap-0.5">
              <Star className="h-3 w-3 fill-yellow-400 text-yellow-400" />
              {recipe.aggregatedrating.toFixed(1)}
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

// Carte simplifiée pour noter une recette aléatoire
function QuickGradeCard({
  recipe,
  onRate,
  onViewDetails,
  submitting,
}: {
  recipe: Recipe
  onRate: (rating: 0 | 1 | 2) => void
  onViewDetails: (recipe: Recipe) => void
  submitting: boolean
}) {
  const imageUrl = getRecipeImageUrl(recipe.images, recipe.recipeid)

  return (
    <Card className="overflow-hidden">
      <div className="relative cursor-pointer" onClick={() => onViewDetails(recipe)}>
        <AspectRatio ratio={16 / 9}>
          <img
            src={imageUrl}
            alt={recipe.name}
            className="h-full w-full object-cover"
            onError={(e) => {
              ;(e.target as HTMLImageElement).src = DEFAULT_FOOD_IMAGE
            }}
          />
        </AspectRatio>
        {recipe.recipecategory && (
          <Badge variant="secondary" className="absolute left-2 bottom-2 text-xs">
            {recipe.recipecategory}
          </Badge>
        )}
      </div>
      <CardContent className="p-3">
        <h4 className="font-medium text-sm line-clamp-1">{recipe.name}</h4>
        <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
          {recipe.totaltime_min && (
            <span className="flex items-center gap-0.5">
              <Clock className="h-3 w-3" />
              {recipe.totaltime_min}min
            </span>
          )}
          {recipe.calories && (
            <span className="flex items-center gap-0.5">
              <Flame className="h-3 w-3" />
              {Math.round(recipe.calories)} cal
            </span>
          )}
        </div>
      </CardContent>
      <CardFooter className="p-3 pt-0">
        <div className="flex justify-center gap-3 w-full">
          <Button
            variant="outline"
            size="sm"
            className="flex-1 border-red-200 hover:bg-red-50 hover:text-red-600 hover:border-red-300"
            onClick={() => onRate(0)}
            disabled={submitting}
          >
            <ThumbsDown className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="flex-1 border-gray-200 hover:bg-gray-50 hover:text-gray-600"
            onClick={() => onRate(1)}
            disabled={submitting}
          >
            <Meh className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="flex-1 border-green-200 hover:bg-green-50 hover:text-green-600 hover:border-green-300"
            onClick={() => onRate(2)}
            disabled={submitting}
          >
            <ThumbsUp className="h-4 w-4" />
          </Button>
        </div>
      </CardFooter>
    </Card>
  )
}

// Skeleton pour les cartes
function CardSkeleton() {
  return (
    <Card className="overflow-hidden">
      <AspectRatio ratio={16 / 9}>
        <Skeleton className="h-full w-full" />
      </AspectRatio>
      <CardContent className="p-3">
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-3 w-1/2 mt-2" />
      </CardContent>
    </Card>
  )
}

export function Dashboard() {
  const { user, profile } = useAuth()
  const navigate = useNavigate()
  const [userProfileId, setUserProfileId] = useState<string | null>(null)

  // Dialog state
  const [selectedRecipe, setSelectedRecipe] = useState<Recipe | null>(null)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [favoriteIds, setFavoriteIds] = useState<Set<number>>(new Set())

  // Favorites
  const [favorites, setFavorites] = useState<Recipe[]>([])
  const [loadingFavorites, setLoadingFavorites] = useState(true)

  // Grading status
  const [recipesToGrade, setRecipesToGrade] = useState<Recipe[]>([])
  const [gradedCount, setGradedCount] = useState(0)
  const [totalToGrade, setTotalToGrade] = useState(0)
  const [loadingGrading, setLoadingGrading] = useState(true)
  const [submittingRating, setSubmittingRating] = useState(false)

  // Recsys status
  const [hasRecsysRecommendations, setHasRecsysRecommendations] = useState(false)
  const [recsysCount, setRecsysCount] = useState(0)
  const [loadingRecsys, setLoadingRecsys] = useState(true)

  // Get time-based greeting
  const getGreeting = () => {
    const hour = new Date().getHours()
    if (hour < 12) return 'Bonjour'
    if (hour < 18) return 'Bon apres-midi'
    return 'Bonsoir'
  }

  // Fetch all data
  useEffect(() => {
    const fetchData = async () => {
      if (!user) return

      try {
        const userProfile = await getUserProfile(user.id)
        if (!userProfile) return

        setUserProfileId(userProfile.id)

        // Fetch favorites
        const { data: favoritesData } = await getUserFavorites(userProfile.id)
        if (favoritesData) {
          setFavorites(favoritesData)
          setFavoriteIds(new Set(favoritesData.map((r) => r.recipeid)))
        }
        setLoadingFavorites(false)

        // Fetch interactions to know what's already graded
        const { data: interactions } = await getUserInteractions(userProfile.id)
        const gradedIds = new Set<number>(interactions?.map((i) => i.recipe_id) || [])
        setGradedCount(gradedIds.size)

        // Fetch filter recommendations (recipes to grade)
        const { data: filterData } = await getUserRecommendations(userProfile.id, 'filter')
        if (filterData && filterData.status === 'ready' && filterData.recipes.length > 0) {
          setTotalToGrade(filterData.recipes.length)
          // Get ungraded recipes for quick grading on dashboard
          const ungraded = filterData.recipes.filter((r) => !gradedIds.has(r.recipeid))
          // Take random 3 for dashboard
          const shuffled = [...ungraded].sort(() => Math.random() - 0.5)
          setRecipesToGrade(shuffled.slice(0, 3))
        }
        setLoadingGrading(false)

        // Fetch recsys recommendations
        const { data: recsysData } = await getUserRecommendations(userProfile.id, 'recsys')
        if (recsysData && recsysData.status === 'ready' && recsysData.recipes.length > 0) {
          setHasRecsysRecommendations(true)
          setRecsysCount(recsysData.recipes.length)
        }
        setLoadingRecsys(false)
      } catch (error) {
        console.error('Error fetching dashboard data:', error)
        setLoadingFavorites(false)
        setLoadingGrading(false)
        setLoadingRecsys(false)
      }
    }

    fetchData()
  }, [user])

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

    if (isFav) {
      setFavorites((prev) => prev.filter((r) => r.recipeid !== recipeId))
    }

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

  const handleQuickRate = async (recipe: Recipe, rating: 0 | 1 | 2) => {
    if (!userProfileId || submittingRating) return

    setSubmittingRating(true)

    const { error } = await createInteraction({
      user_id: userProfileId,
      recipe_id: recipe.recipeid,
      rating,
    })

    if (error) {
      toast.error("Erreur lors de l'envoi de la note")
      setSubmittingRating(false)
      return
    }

    // Remove rated recipe from list
    setRecipesToGrade((prev) => prev.filter((r) => r.recipeid !== recipe.recipeid))
    setGradedCount((prev) => prev + 1)
    setSubmittingRating(false)

    toast.success(
      rating === 2 ? "Tu aimes cette recette !" : rating === 0 ? 'Note enregistree' : 'Note enregistree'
    )
  }

  const openRecipeDetail = (recipe: Recipe) => {
    setSelectedRecipe(recipe)
    setDialogOpen(true)
  }

  const gradingProgress = totalToGrade > 0 ? Math.round((gradedCount / totalToGrade) * 100) : 0
  const allGraded = totalToGrade > 0 && gradedCount >= totalToGrade

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
                <BreadcrumbPage>Dashboard</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </header>

        <div className="flex flex-1 flex-col gap-6 p-4 md:p-6">
          {/* Welcome Card */}
          <Card className="bg-linear-to-r from-primary/10 via-primary/5 to-transparent border-primary/20">
            <CardHeader>
              <CardTitle className="text-2xl">
                {getGreeting()}, {profile?.first_name || 'Gourmand'} !
              </CardTitle>
              <CardDescription className="text-base">
                Pret a decouvrir de nouvelles recettes delicieuses ?
              </CardDescription>
            </CardHeader>
          </Card>

          {/* Status Cards */}
          <div className="grid gap-4 md:grid-cols-2">
            {/* Grading Status Card */}
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Filter className="h-5 w-5 text-primary" />
                    <CardTitle className="text-base">Recettes a noter</CardTitle>
                  </div>
                  {loadingGrading ? (
                    <Skeleton className="h-6 w-16" />
                  ) : allGraded ? (
                    <Badge className="bg-green-100 text-green-700 border-green-200">
                      <CheckCircle2 className="h-3 w-3 mr-1" />
                      Termine
                    </Badge>
                  ) : (
                    <Badge variant="secondary">{gradingProgress}%</Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent className="pb-3">
                {loadingGrading ? (
                  <Skeleton className="h-4 w-full" />
                ) : (
                  <>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary transition-all duration-500"
                        style={{ width: `${gradingProgress}%` }}
                      />
                    </div>
                    <p className="text-sm text-muted-foreground mt-2">
                      {allGraded
                        ? 'Tu as note toutes les recettes disponibles !'
                        : `${gradedCount} sur ${totalToGrade} recettes notees`}
                    </p>
                  </>
                )}
              </CardContent>
              <CardFooter className="pt-0">
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full"
                  onClick={() => navigate('/recommendations?tab=filter')}
                >
                  {allGraded ? 'Demander plus de recettes' : 'Continuer a noter'}
                  <ArrowRight className="h-4 w-4 ml-2" />
                </Button>
              </CardFooter>
            </Card>

            {/* Recsys Status Card */}
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-primary" />
                    <CardTitle className="text-base">Recommandations IA</CardTitle>
                  </div>
                  {loadingRecsys ? (
                    <Skeleton className="h-6 w-16" />
                  ) : hasRecsysRecommendations ? (
                    <Badge className="bg-green-100 text-green-700 border-green-200">
                      <CheckCircle2 className="h-3 w-3 mr-1" />
                      Disponibles
                    </Badge>
                  ) : (
                    <Badge variant="secondary" className="bg-orange-100 text-orange-700 border-orange-200">
                      <AlertCircle className="h-3 w-3 mr-1" />
                      En attente
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent className="pb-3">
                {loadingRecsys ? (
                  <Skeleton className="h-4 w-full" />
                ) : hasRecsysRecommendations ? (
                  <p className="text-sm text-muted-foreground">
                    {recsysCount} recettes personnalisees t'attendent !
                  </p>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    Note plus de recettes pour debloquer tes recommandations personnalisees.
                  </p>
                )}
              </CardContent>
              <CardFooter className="pt-0">
                <Button
                  variant={hasRecsysRecommendations ? 'default' : 'outline'}
                  size="sm"
                  className="w-full"
                  onClick={() => navigate('/recommendations?tab=recsys')}
                  disabled={!hasRecsysRecommendations}
                >
                  {hasRecsysRecommendations ? 'Voir mes recommandations' : 'Pas encore disponible'}
                  {hasRecsysRecommendations && <ArrowRight className="h-4 w-4 ml-2" />}
                </Button>
              </CardFooter>
            </Card>
          </div>

          {/* Quick Grading Section */}
          {!allGraded && recipesToGrade.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold flex items-center gap-2">
                    <ChefHat className="h-5 w-5 text-primary" />
                    Note ces recettes
                  </h2>
                  <p className="text-sm text-muted-foreground">
                    Aide-nous a mieux te connaitre
                  </p>
                </div>
                <Button variant="ghost" size="sm" onClick={() => navigate('/recommendations?tab=filter')}>
                  Voir tout
                  <ArrowRight className="h-4 w-4 ml-1" />
                </Button>
              </div>

              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {recipesToGrade.map((recipe) => (
                  <QuickGradeCard
                    key={recipe.recipeid}
                    recipe={recipe}
                    onRate={(rating) => handleQuickRate(recipe, rating)}
                    onViewDetails={openRecipeDetail}
                    submitting={submittingRating}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Favorites Section */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold flex items-center gap-2">
                  <Heart className="h-5 w-5 text-red-500" />
                  Mes favoris
                </h2>
                <p className="text-sm text-muted-foreground">
                  {favorites.length > 0
                    ? `${favorites.length} recette${favorites.length > 1 ? 's' : ''} sauvegardee${favorites.length > 1 ? 's' : ''}`
                    : 'Aucun favori pour le moment'}
                </p>
              </div>
              {favorites.length > 4 && (
                <Button variant="ghost" size="sm" onClick={() => navigate('/favorites')}>
                  Voir tout
                  <ArrowRight className="h-4 w-4 ml-1" />
                </Button>
              )}
            </div>

            {loadingFavorites ? (
              <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
                {[...Array(4)].map((_, i) => (
                  <CardSkeleton key={i} />
                ))}
              </div>
            ) : favorites.length > 0 ? (
              <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
                {favorites.slice(0, 4).map((recipe) => (
                  <FavoriteCard
                    key={recipe.recipeid}
                    recipe={recipe}
                    onViewDetails={openRecipeDetail}
                    onToggleFavorite={handleToggleFavorite}
                  />
                ))}
              </div>
            ) : (
              <Card className="border-dashed">
                <CardContent className="flex flex-col items-center justify-center py-8 text-center">
                  <Heart className="h-10 w-10 text-muted-foreground/50 mb-3" />
                  <p className="text-sm text-muted-foreground">
                    Tu n'as pas encore de recettes favorites.
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Explore les recommandations et clique sur le coeur pour sauvegarder tes recettes preferees !
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    className="mt-4"
                    onClick={() => navigate('/recommendations?tab=recsys')}
                  >
                    Explorer les recommandations
                  </Button>
                </CardContent>
              </Card>
            )}
          </div>
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
