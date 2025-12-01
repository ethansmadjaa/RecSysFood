import { useEffect, useState } from 'react'
import { motion, useMotionValue, useTransform, AnimatePresence, animate } from 'framer-motion'
import type { PanInfo } from 'framer-motion'
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
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Heart, ChefHat, Clock, Star, Loader2, Flame, Drumstick, Leaf, RefreshCw, Users, Calendar, AlertTriangle, Utensils, Timer, ThumbsDown, ThumbsUp, Meh } from 'lucide-react'
import { getUserRecommendations, triggerRecommendations, type Recipe, type RecommendationsResponse } from '@/lib/api/recommendations'
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
      const gradedIds = new Set<number>(interactions?.map(i => i.recipe_id) || [])

      const { data } = await getUserRecommendations(userProfile.id)

      console.log('data', data)

      if (data) {
        setStatus(data.status)
        if (data.status === 'ready' && data.recipes.length > 0) {
          setRecommendations(data.recipes)

          // Filter out already graded recipes
          const ungraded = data.recipes.filter(r => !gradedIds.has(r.recipeid))
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
    setFavoriteIds(prev => {
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
      setFavoriteIds(prev => {
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
    if (!userProfileId || currentRecipeIndex >= recipesToGrade.length || submittingRating) return

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
      toast.error('Erreur lors de l\'envoi de la note')
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

  // Curated list of high-quality food images from Unsplash
  const foodImageIds = [
    'photo-1546069901-ba9599a7e63c', // colorful salad bowl
    'photo-1567620905732-2d1ec7ab7445', // pancakes
    'photo-1565299624946-b28f40a0ae38', // pizza
    'photo-1540189549336-e6e99c3679fe', // food platter
    'photo-1565958011703-44f9829ba187', // dessert
    'photo-1621996346565-e3dbc646d9a9', // pasta
    'photo-1504674900247-0877df9cc836', // grilled food
    'photo-1512621776951-a57141f2eefd', // healthy bowl
    'photo-1473093295043-cdd812d0e601', // pasta dish
    'photo-1476224203421-9ac39bcb3327', // breakfast
    'photo-1484723091739-30a097e8f929', // french toast
    'photo-1432139555190-58524dae6a55', // salad
    'photo-1529042410759-befb1204b468', // ramen
    'photo-1414235077428-338989a2e8c0', // fine dining
    'photo-1490645935967-10de6ba17061', // food spread
    'photo-1498837167922-ddd27525d352', // vegetables
    'photo-1455619452474-d2be8b1e70cd', // asian food
    'photo-1476718406336-bb5a9690ee2a', // burger
    'photo-1499028344343-cd173ffc68a9', // tacos
    'photo-1547592180-85f173990554', // breakfast bowl
    'photo-1493770348161-369560ae357d', // healthy food
    'photo-1506354666786-959d6d497f1a', // smoothie bowl
    'photo-1551183053-bf91a1d81141', // sushi
    'photo-1559847844-5315695dadae', // curry
    'photo-1574484284002-952d92456975', // indian food
    'photo-1585937421612-70a008356fbe', // mediterranean
    'photo-1563379926898-05f4575a45d8', // pasta carbonara
    'photo-1569718212165-3a8278d5f624', // fried rice
    'photo-1604908176997-125f25cc6f3d', // steak
    'photo-1594007654729-407eedc4be65', // noodles
  ]

  // Get a consistent but varied image based on recipe ID
  const getUnsplashFoodImage = (recipeId: number) => {
    const imageIndex = recipeId % foodImageIds.length
    const imageId = foodImageIds[imageIndex]
    return `https://images.unsplash.com/${imageId}?w=800&h=500&fit=crop&auto=format`
  }

  const openRecipeDetail = (recipe: Recipe) => {
    setSelectedRecipe(recipe)
    setDialogOpen(true)
  }

  const RecipeCard = ({ recipe }: { recipe: Recipe }) => {
    const originalImageUrl = recipe.images?.[0] || null
    const fallbackImageUrl = getUnsplashFoodImage(recipe.recipeid)
    const isFavorite = favoriteIds.has(recipe.recipeid)

    return (
      <Card className="group overflow-hidden transition-all hover:shadow-lg">
        <div className="relative">
          <AspectRatio ratio={16 / 10}>
            <img
              src={originalImageUrl || fallbackImageUrl}
              alt={recipe.name}
              className="h-full w-full object-cover transition-transform group-hover:scale-105"
              onError={(e) => {
                (e.target as HTMLImageElement).src = 'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=800&auto=format&fit=crop'
              }}
            />
          </AspectRatio>
          <Button
            variant="secondary"
            size="icon"
            className={`absolute right-2 top-2 h-8 w-8 rounded-full transition-opacity ${isFavorite ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}
            onClick={(e) => {
              e.stopPropagation()
              handleToggleFavorite(recipe.recipeid)
            }}
          >
            <Heart className={`h-4 w-4 ${isFavorite ? 'fill-red-500 text-red-500' : ''}`} />
          </Button>
          {recipe.aggregatedrating && recipe.aggregatedrating >= 4.5 && (
            <Badge className="absolute left-2 top-2 bg-yellow-500 text-yellow-950">
              <Star className="mr-1 h-3 w-3 fill-current" />
              Top note
            </Badge>
          )}
          {recipe.recipecategory && (
            <Badge variant="secondary" className="absolute left-2 bottom-2">
              {recipe.recipecategory}
            </Badge>
          )}
        </div>

        <CardHeader className="pb-2">
          <CardTitle className="line-clamp-2 text-base">{recipe.name}</CardTitle>
          {recipe.description && (
            <p className="text-xs text-muted-foreground line-clamp-2 mt-1">{recipe.description}</p>
          )}
          <CardDescription className="flex items-center gap-3 text-xs mt-2">
            {recipe.totaltime_min && (
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {recipe.totaltime_min} min
              </span>
            )}
            {recipe.aggregatedrating && (
              <span className="flex items-center gap-1">
                <Star className="h-3 w-3 fill-yellow-400 text-yellow-400" />
                {recipe.aggregatedrating.toFixed(1)}
              </span>
            )}
            {recipe.reviewcount && (
              <span className="text-muted-foreground">
                ({Math.round(recipe.reviewcount)} avis)
              </span>
            )}
            {recipe.recipeservings && (
              <span className="flex items-center gap-1">
                <Users className="h-3 w-3" />
                {recipe.recipeservings} pers.
              </span>
            )}
          </CardDescription>
        </CardHeader>

        <CardContent className="pb-2">
          <div className="flex flex-wrap gap-1.5">
            {recipe.is_vegan && (
              <Badge variant="outline" className="text-xs bg-green-50 text-green-700 border-green-200">
                <Leaf className="mr-1 h-3 w-3" />
                Vegan
              </Badge>
            )}
            {recipe.is_vegetarian && !recipe.is_vegan && (
              <Badge variant="outline" className="text-xs bg-green-50 text-green-700 border-green-200">
                <Leaf className="mr-1 h-3 w-3" />
                Vegetarien
              </Badge>
            )}
            {recipe.calories && (
              <Badge variant="outline" className="text-xs">
                <Flame className="mr-1 h-3 w-3" />
                {Math.round(recipe.calories)} cal
              </Badge>
            )}
            {recipe.proteincontent && recipe.proteincontent > 20 && (
              <Badge variant="outline" className="text-xs bg-blue-50 text-blue-700 border-blue-200">
                <Drumstick className="mr-1 h-3 w-3" />
                Riche en proteines
              </Badge>
            )}
          </div>
        </CardContent>

        <CardFooter className="pt-2">
          <Button variant="default" className="w-full" size="sm" onClick={() => openRecipeDetail(recipe)}>
            Voir la recette
          </Button>
        </CardFooter>
      </Card>
    )
  }

  const RecipeDetailDialog = ({ recipe }: { recipe: Recipe }) => {
    const originalImageUrl = recipe.images?.[0] || null
    const fallbackImageUrl = getUnsplashFoodImage(recipe.recipeid)
    const isFavorite = favoriteIds.has(recipe.recipeid)

    // Combine ingredients with quantities
    const ingredients = recipe.recipeingredientparts?.map((part, index) => ({
      name: part,
      quantity: recipe.recipeingredientquantities?.[index] || ''
    })) || []

    // Allergens/dietary warnings
    const warnings = []
    if (recipe.contains_nuts) warnings.push('Noix')
    if (recipe.contains_dairy) warnings.push('Produits laitiers')
    if (recipe.contains_egg) warnings.push('Oeufs')
    if (recipe.contains_fish) warnings.push('Poisson')
    if (recipe.contains_soy) warnings.push('Soja')
    if (recipe.contains_gluten) warnings.push('Gluten')
    if (recipe.contains_pork) warnings.push('Porc')
    if (recipe.contains_alcohol) warnings.push('Alcool')

    return (
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader className="flex flex-row items-start justify-between gap-4">
          <div className="flex-1">
            <DialogTitle className="text-xl">{recipe.name}</DialogTitle>
            {recipe.authorname && (
              <p className="text-sm text-muted-foreground">Par {recipe.authorname}</p>
            )}
          </div>
          <Button
            variant={isFavorite ? "default" : "outline"}
            size="sm"
            onClick={() => handleToggleFavorite(recipe.recipeid)}
            className="shrink-0"
          >
            <Heart className={`h-4 w-4 mr-2 ${isFavorite ? 'fill-current' : ''}`} />
            {isFavorite ? 'Favori' : 'Ajouter aux favoris'}
          </Button>
        </DialogHeader>

        <ScrollArea className="flex-1 pr-4">
          <div className="space-y-6">
            {/* Image */}
            <div className="rounded-lg overflow-hidden bg-muted">
              <img
                src={originalImageUrl || fallbackImageUrl}
                alt={recipe.name}
                className="w-full h-auto max-h-[250px] object-cover"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = 'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=800&auto=format&fit=crop'
                }}
              />
            </div>

            {/* Infos principales */}
            <div className="space-y-4">
                {recipe.description && (
                  <p className="text-sm text-muted-foreground">{recipe.description}</p>
                )}

                {/* Temps et portions */}
                <div className="grid grid-cols-2 gap-2">
                  {recipe.preptime_min && (
                    <div className="flex items-center gap-2 p-2 bg-muted rounded-lg">
                      <Timer className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <p className="text-xs text-muted-foreground">Preparation</p>
                        <p className="text-sm font-medium">{recipe.preptime_min} min</p>
                      </div>
                    </div>
                  )}
                  {recipe.cooktime_min && (
                    <div className="flex items-center gap-2 p-2 bg-muted rounded-lg">
                      <Utensils className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <p className="text-xs text-muted-foreground">Cuisson</p>
                        <p className="text-sm font-medium">{recipe.cooktime_min} min</p>
                      </div>
                    </div>
                  )}
                  {recipe.totaltime_min && (
                    <div className="flex items-center gap-2 p-2 bg-muted rounded-lg">
                      <Clock className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <p className="text-xs text-muted-foreground">Temps total</p>
                        <p className="text-sm font-medium">{recipe.totaltime_min} min</p>
                      </div>
                    </div>
                  )}
                  {recipe.recipeservings && (
                    <div className="flex items-center gap-2 p-2 bg-muted rounded-lg">
                      <Users className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <p className="text-xs text-muted-foreground">Portions</p>
                        <p className="text-sm font-medium">{recipe.recipeservings} {recipe.recipeyield || 'portions'}</p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Note et avis */}
                {recipe.aggregatedrating && (
                  <div className="flex items-center gap-2">
                    <div className="flex items-center">
                      {[1, 2, 3, 4, 5].map((star) => (
                        <Star
                          key={star}
                          className={`h-4 w-4 ${star <= recipe.aggregatedrating! ? 'fill-yellow-400 text-yellow-400' : 'text-gray-300'}`}
                        />
                      ))}
                    </div>
                    <span className="text-sm font-medium">{recipe.aggregatedrating.toFixed(1)}</span>
                    {recipe.reviewcount && (
                      <span className="text-sm text-muted-foreground">({Math.round(recipe.reviewcount)} avis)</span>
                    )}
                  </div>
                )}

                {/* Tags */}
                <div className="flex flex-wrap gap-1.5">
                  {recipe.recipecategory && (
                    <Badge variant="secondary">{recipe.recipecategory}</Badge>
                  )}
                  {recipe.is_vegan && (
                    <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                      <Leaf className="mr-1 h-3 w-3" />
                      Vegan
                    </Badge>
                  )}
                  {recipe.is_vegetarian && !recipe.is_vegan && (
                    <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                      <Leaf className="mr-1 h-3 w-3" />
                      Vegetarien
                    </Badge>
                  )}
                  {recipe.is_breakfast_brunch && (
                    <Badge variant="outline">Petit-dejeuner</Badge>
                  )}
                  {recipe.is_dessert && (
                    <Badge variant="outline">Dessert</Badge>
                  )}
                </div>

                {/* Avertissements allergenes */}
                {warnings.length > 0 && (
                  <div className="p-3 bg-orange-50 border border-orange-200 rounded-lg">
                    <div className="flex items-center gap-2 text-orange-700">
                      <AlertTriangle className="h-4 w-4" />
                      <span className="text-sm font-medium">Contient:</span>
                    </div>
                    <p className="text-sm text-orange-600 mt-1">{warnings.join(', ')}</p>
                  </div>
                )}
            </div>

            {/* Tabs pour ingredients, instructions, nutrition */}
            <Tabs defaultValue="ingredients" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="ingredients">Ingredients</TabsTrigger>
                <TabsTrigger value="instructions">Instructions</TabsTrigger>
                <TabsTrigger value="nutrition">Nutrition</TabsTrigger>
              </TabsList>

              <TabsContent value="ingredients" className="mt-4">
                {ingredients.length > 0 ? (
                  <ul className="space-y-2">
                    {ingredients.map((ingredient, index) => (
                      <li key={index} className="flex items-center gap-2 p-2 bg-muted rounded-lg">
                        <span className="font-medium text-sm">{ingredient.quantity}</span>
                        <span className="text-sm">{ingredient.name}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-muted-foreground">Ingredients non disponibles</p>
                )}
              </TabsContent>

              <TabsContent value="instructions" className="mt-4">
                {(() => {
                  let instructions: string[] = []
                  if (recipe.recipeinstructions) {
                    if (Array.isArray(recipe.recipeinstructions)) {
                      instructions = recipe.recipeinstructions
                    } else if (typeof recipe.recipeinstructions === 'string') {
                      const str = recipe.recipeinstructions.trim()
                      // Check if it's a Python-style array string: ['item1', 'item2']
                      if (str.startsWith('[') && str.endsWith(']')) {
                        // Extract content between brackets and split by "', '"
                        const content = str.slice(2, -2) // Remove [' and ']
                        instructions = content.split("', '").map((s: string) => s.trim())
                      } else {
                        instructions = [str]
                      }
                    }
                  }
                  return instructions.length > 0 ? (
                    <ol className="space-y-3">
                      {instructions.map((instruction, index) => (
                        <li key={index} className="flex gap-3 p-3 bg-muted rounded-lg">
                          <span className="shrink-0 w-6 h-6 rounded-full bg-primary text-primary-foreground text-sm flex items-center justify-center">
                            {index + 1}
                          </span>
                          <span className="text-sm">{instruction}</span>
                        </li>
                      ))}
                    </ol>
                  ) : (
                    <p className="text-sm text-muted-foreground">Instructions non disponibles</p>
                  )
                })()}
              </TabsContent>

              <TabsContent value="nutrition" className="mt-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {recipe.calories != null && recipe.calories > 0 && (
                    <div className="p-3 bg-muted rounded-lg text-center">
                      <Flame className="h-5 w-5 mx-auto text-orange-500" />
                      <p className="text-lg font-bold mt-1">{Math.round(recipe.calories)}</p>
                      <p className="text-xs text-muted-foreground">Calories</p>
                    </div>
                  )}
                  {recipe.proteincontent != null && recipe.proteincontent > 0 && (
                    <div className="p-3 bg-muted rounded-lg text-center">
                      <Drumstick className="h-5 w-5 mx-auto text-red-500" />
                      <p className="text-lg font-bold mt-1">{Math.round(recipe.proteincontent)}g</p>
                      <p className="text-xs text-muted-foreground">Proteines</p>
                    </div>
                  )}
                  {recipe.carbohydratecontent != null && recipe.carbohydratecontent > 0 && (
                    <div className="p-3 bg-muted rounded-lg text-center">
                      <div className="h-5 w-5 mx-auto text-yellow-500 font-bold text-sm">C</div>
                      <p className="text-lg font-bold mt-1">{Math.round(recipe.carbohydratecontent)}g</p>
                      <p className="text-xs text-muted-foreground">Glucides</p>
                    </div>
                  )}
                  {recipe.fatcontent != null && recipe.fatcontent > 0 && (
                    <div className="p-3 bg-muted rounded-lg text-center">
                      <div className="h-5 w-5 mx-auto text-blue-500 font-bold text-sm">F</div>
                      <p className="text-lg font-bold mt-1">{Math.round(recipe.fatcontent)}g</p>
                      <p className="text-xs text-muted-foreground">Lipides</p>
                    </div>
                  )}
                  {recipe.fibercontent != null && recipe.fibercontent > 0 && (
                    <div className="p-3 bg-muted rounded-lg text-center">
                      <Leaf className="h-5 w-5 mx-auto text-green-500" />
                      <p className="text-lg font-bold mt-1">{Math.round(recipe.fibercontent)}g</p>
                      <p className="text-xs text-muted-foreground">Fibres</p>
                    </div>
                  )}
                  {recipe.sugarcontent != null && recipe.sugarcontent > 0 && (
                    <div className="p-3 bg-muted rounded-lg text-center">
                      <div className="h-5 w-5 mx-auto text-pink-500 font-bold text-sm">S</div>
                      <p className="text-lg font-bold mt-1">{Math.round(recipe.sugarcontent)}g</p>
                      <p className="text-xs text-muted-foreground">Sucres</p>
                    </div>
                  )}
                  {recipe.sodiumcontent != null && recipe.sodiumcontent > 0 && (
                    <div className="p-3 bg-muted rounded-lg text-center">
                      <div className="h-5 w-5 mx-auto text-gray-500 font-bold text-sm">Na</div>
                      <p className="text-lg font-bold mt-1">{Math.round(recipe.sodiumcontent)}mg</p>
                      <p className="text-xs text-muted-foreground">Sodium</p>
                    </div>
                  )}
                  {recipe.cholesterolcontent != null && recipe.cholesterolcontent > 0 && (
                    <div className="p-3 bg-muted rounded-lg text-center">
                      <div className="h-5 w-5 mx-auto text-purple-500 font-bold text-sm">Ch</div>
                      <p className="text-lg font-bold mt-1">{Math.round(recipe.cholesterolcontent)}mg</p>
                      <p className="text-xs text-muted-foreground">Cholesterol</p>
                    </div>
                  )}
                </div>
              </TabsContent>
            </Tabs>

            {/* Keywords */}
            {recipe.keywords && recipe.keywords.length > 0 && (
              <div className="pt-4 border-t">
                <p className="text-sm font-medium mb-2">Mots-cles</p>
                <div className="flex flex-wrap gap-1">
                  {recipe.keywords.map((keyword, index) => (
                    <Badge key={index} variant="outline" className="text-xs">
                      {keyword}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Date de publication */}
            {recipe.datepublished && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground pt-2">
                <Calendar className="h-3 w-3" />
                Publie le {new Date(recipe.datepublished).toLocaleDateString('fr-FR')}
              </div>
            )}
          </div>
        </ScrollArea>
      </DialogContent>
    )
  }

  const GradingCard = ({ recipe, onRate }: { recipe: Recipe; onRate: (rating: 0 | 1 | 2) => void }) => {
    const originalImageUrl = recipe.images?.[0] || null
    const fallbackImageUrl = getUnsplashFoodImage(recipe.recipeid)

    // Motion values for drag
    const x = useMotionValue(0)
    const y = useMotionValue(0)

    // Transform values based on drag position
    const rotate = useTransform(x, [-300, 0, 300], [-25, 0, 25])
    const likeOpacity = useTransform(x, [0, 100, 200], [0, 0.5, 1])
    const dislikeOpacity = useTransform(x, [-200, -100, 0], [1, 0.5, 0])
    const mehOpacity = useTransform(y, [0, 100, 200], [0, 0.5, 1])

    // Scale effect while dragging
    const scale = useTransform(
      x,
      [-300, -150, 0, 150, 300],
      [0.95, 0.98, 1, 0.98, 0.95]
    )

    // Swipe threshold
    const SWIPE_THRESHOLD = 120
    const SWIPE_Y_THRESHOLD = 100

    const handleDragEnd = (_: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
      const xOffset = info.offset.x
      const yOffset = info.offset.y
      const xVelocity = info.velocity.x
      const yVelocity = info.velocity.y

      // Check for vertical swipe (meh) first
      if (yOffset > SWIPE_Y_THRESHOLD || yVelocity > 500) {
        onRate(1) // Meh
      }
      // Check for horizontal swipes
      else if (xOffset > SWIPE_THRESHOLD || xVelocity > 500) {
        onRate(2) // Like
      } else if (xOffset < -SWIPE_THRESHOLD || xVelocity < -500) {
        onRate(0) // Dislike
      }
    }

    // Handle button click with animation
    const handleButtonClick = (rating: 0 | 1 | 2) => {
      const targetX = rating === 2 ? 500 : rating === 0 ? -500 : 0
      const targetY = rating === 1 ? 400 : 0

      // Animate the card out
      animate(x, targetX, { duration: 0.3, ease: 'easeOut' })
      animate(y, targetY, { duration: 0.3, ease: 'easeOut' })

      // Call onRate after animation starts
      setTimeout(() => {
        onRate(rating)
      }, 250)
    }

    return (
      <motion.div
        className="w-full cursor-grab active:cursor-grabbing"
        style={{ x, y, rotate, scale }}
        drag
        dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
        dragElastic={0.9}
        onDragEnd={handleDragEnd}
        whileDrag={{ cursor: 'grabbing' }}
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{
          x: swipeDirection === 'right' ? 500 : swipeDirection === 'left' ? -500 : 0,
          y: swipeDirection === 'down' ? 500 : 0,
          rotate: swipeDirection === 'right' ? 20 : swipeDirection === 'left' ? -20 : 0,
          opacity: 0,
          transition: { duration: 0.3, ease: 'easeOut' }
        }}
        transition={{ type: 'spring', stiffness: 300, damping: 25 }}
      >
        <Card className="w-full overflow-hidden shadow-xl relative">
          {/* Like overlay */}
          <motion.div
            className="absolute inset-0 bg-green-500/20 z-10 pointer-events-none flex items-center justify-center"
            style={{ opacity: likeOpacity }}
          >
            <div className="bg-green-500 text-white px-6 py-3 rounded-xl rotate-[-15deg] border-4 border-green-600 shadow-lg">
              <div className="flex items-center gap-2">
                <ThumbsUp className="h-8 w-8" />
                <span className="text-2xl font-bold">J'AIME</span>
              </div>
            </div>
          </motion.div>

          {/* Dislike overlay */}
          <motion.div
            className="absolute inset-0 bg-red-500/20 z-10 pointer-events-none flex items-center justify-center"
            style={{ opacity: dislikeOpacity }}
          >
            <div className="bg-red-500 text-white px-6 py-3 rounded-xl rotate-15 border-4 border-red-600 shadow-lg">
              <div className="flex items-center gap-2">
                <ThumbsDown className="h-8 w-8" />
                <span className="text-2xl font-bold">NOPE</span>
              </div>
            </div>
          </motion.div>

          {/* Meh overlay */}
          <motion.div
            className="absolute inset-0 bg-gray-500/20 z-10 pointer-events-none flex items-center justify-center"
            style={{ opacity: mehOpacity }}
          >
            <div className="bg-gray-500 text-white px-6 py-3 rounded-xl border-4 border-gray-600 shadow-lg">
              <div className="flex items-center gap-2">
                <Meh className="h-8 w-8" />
                <span className="text-2xl font-bold">BOF</span>
              </div>
            </div>
          </motion.div>

          <div className="relative">
            <AspectRatio ratio={4 / 3}>
              <img
                src={originalImageUrl || fallbackImageUrl}
                alt={recipe.name}
                className="h-full w-full object-cover"
                draggable={false}
                onError={(e) => {
                  (e.target as HTMLImageElement).src = 'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=800&auto=format&fit=crop'
                }}
              />
            </AspectRatio>
            {recipe.recipecategory && (
              <Badge variant="secondary" className="absolute left-3 bottom-3">
                {recipe.recipecategory}
              </Badge>
            )}
            {recipe.aggregatedrating && recipe.aggregatedrating >= 4.5 && (
              <Badge className="absolute right-3 top-3 bg-yellow-500 text-yellow-950">
                <Star className="mr-1 h-3 w-3 fill-current" />
                Top note
              </Badge>
            )}
          </div>

          <CardHeader className="pb-3">
            <CardTitle className="text-xl line-clamp-2">{recipe.name}</CardTitle>
            {recipe.description && (
              <p className="text-sm text-muted-foreground line-clamp-2 mt-1">{recipe.description}</p>
            )}
            <CardDescription className="flex flex-wrap items-center gap-3 text-sm mt-2">
              {recipe.totaltime_min && (
                <span className="flex items-center gap-1">
                  <Clock className="h-4 w-4" />
                  {recipe.totaltime_min} min
                </span>
              )}
              {recipe.aggregatedrating && (
                <span className="flex items-center gap-1">
                  <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                  {recipe.aggregatedrating.toFixed(1)}
                </span>
              )}
              {recipe.recipeservings && (
                <span className="flex items-center gap-1">
                  <Users className="h-4 w-4" />
                  {recipe.recipeservings} pers.
                </span>
              )}
            </CardDescription>
          </CardHeader>

          <CardContent className="pb-3">
            <div className="flex flex-wrap gap-2">
              {recipe.is_vegan && (
                <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                  <Leaf className="mr-1 h-3 w-3" />
                  Vegan
                </Badge>
              )}
              {recipe.is_vegetarian && !recipe.is_vegan && (
                <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                  <Leaf className="mr-1 h-3 w-3" />
                  Vegetarien
                </Badge>
              )}
              {recipe.calories && (
                <Badge variant="outline">
                  <Flame className="mr-1 h-3 w-3" />
                  {Math.round(recipe.calories)} cal
                </Badge>
              )}
              {recipe.proteincontent && recipe.proteincontent > 20 && (
                <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                  <Drumstick className="mr-1 h-3 w-3" />
                  Riche en proteines
                </Badge>
              )}
            </div>
          </CardContent>

          <CardFooter className="flex flex-col gap-4 pt-4 pb-6">
            <Button
              variant="ghost"
              size="sm"
              className="text-muted-foreground hover:text-foreground"
              onClick={(e) => {
                e.stopPropagation()
                openRecipeDetail(recipe)
              }}
            >
              <Utensils className="h-4 w-4 mr-2" />
              Voir les details
            </Button>

            {/* Swipe hint text */}
            <p className="text-xs text-muted-foreground text-center">
              Swipe la carte ou utilise les boutons
            </p>

            <div className="flex justify-center items-center gap-6">
              {/* Dislike button */}
              <motion.button
                className="h-16 w-16 rounded-full border-3 border-red-400 bg-white flex items-center justify-center shadow-lg shadow-red-100 disabled:opacity-50 disabled:cursor-not-allowed"
                whileHover={{
                  scale: 1.15,
                  boxShadow: '0 10px 30px -10px rgba(239, 68, 68, 0.5)',
                  borderColor: 'rgb(239, 68, 68)'
                }}
                whileTap={{ scale: 0.9 }}
                onClick={() => handleButtonClick(0)}
                disabled={submittingRating}
                transition={{ type: 'spring', stiffness: 400, damping: 17 }}
              >
                <motion.div
                  whileHover={{ rotate: -15 }}
                  transition={{ type: 'spring', stiffness: 300 }}
                >
                  <ThumbsDown className="h-7 w-7 text-red-500" />
                </motion.div>
              </motion.button>

              {/* Meh button */}
              <motion.button
                className="h-14 w-14 rounded-full border-2 border-gray-300 bg-white flex items-center justify-center shadow-md shadow-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
                whileHover={{
                  scale: 1.1,
                  boxShadow: '0 8px 25px -8px rgba(107, 114, 128, 0.4)',
                  borderColor: 'rgb(107, 114, 128)'
                }}
                whileTap={{ scale: 0.9 }}
                onClick={() => handleButtonClick(1)}
                disabled={submittingRating}
                transition={{ type: 'spring', stiffness: 400, damping: 17 }}
              >
                <Meh className="h-6 w-6 text-gray-500" />
              </motion.button>

              {/* Like button */}
              <motion.button
                className="h-16 w-16 rounded-full border-3 border-green-400 bg-white flex items-center justify-center shadow-lg shadow-green-100 disabled:opacity-50 disabled:cursor-not-allowed"
                whileHover={{
                  scale: 1.15,
                  boxShadow: '0 10px 30px -10px rgba(34, 197, 94, 0.5)',
                  borderColor: 'rgb(34, 197, 94)'
                }}
                whileTap={{ scale: 0.9 }}
                onClick={() => handleButtonClick(2)}
                disabled={submittingRating}
                transition={{ type: 'spring', stiffness: 400, damping: 17 }}
              >
                <motion.div
                  whileHover={{ rotate: 15 }}
                  transition={{ type: 'spring', stiffness: 300 }}
                >
                  <ThumbsUp className="h-7 w-7 text-green-500" />
                </motion.div>
              </motion.button>
            </div>
          </CardFooter>
        </Card>
      </motion.div>
    )
  }

  // Wrapper component for the grading section with AnimatePresence
  const GradingSection = () => {
    const currentRecipe = recipesToGrade[currentRecipeIndex]
    if (!currentRecipe) return null

    return (
      <div className="flex flex-col items-center justify-center w-full max-w-lg mx-auto">
        {/* Progress indicator */}
        <div className="w-full mb-4">
          <div className="flex justify-between text-sm text-muted-foreground mb-2">
            <span>Recette {currentRecipeIndex + 1} sur {recipesToGrade.length}</span>
            <span>{Math.round(((currentRecipeIndex) / recipesToGrade.length) * 100)}%</span>
          </div>
          <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
            <motion.div
              className="bg-primary h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${(currentRecipeIndex / recipesToGrade.length) * 100}%` }}
              transition={{ duration: 0.3, ease: 'easeOut' }}
            />
          </div>
        </div>

        {/* Card stack with AnimatePresence */}
        <div className="relative w-full h-[600px]">
          <AnimatePresence mode="wait">
            <GradingCard
              key={currentRecipe.recipeid}
              recipe={currentRecipe}
              onRate={handleRating}
            />
          </AnimatePresence>
        </div>

      </div>
    )
  }

  const GradingComplete = () => (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 200, damping: 20 }}
    >
      <Card className="w-full max-w-lg mx-auto overflow-hidden">
        <CardContent className="flex flex-col items-center justify-center py-12">
          <motion.div
            className="rounded-full bg-green-100 p-5 mb-6"
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ type: 'spring', stiffness: 200, damping: 15, delay: 0.2 }}
          >
            <motion.div
              animate={{
                scale: [1, 1.2, 1],
              }}
              transition={{ repeat: 2, duration: 0.4, delay: 0.5 }}
            >
              <ThumbsUp className="h-14 w-14 text-green-600" />
            </motion.div>
          </motion.div>
          <motion.h2
            className="text-2xl font-bold text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            Merci pour tes notes !
          </motion.h2>
          <motion.p
            className="text-muted-foreground text-center mt-2 max-w-md"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            Tes preferences ont ete enregistrees. Nous allons ameliorer tes recommandations.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Button className="mt-6" size="lg" onClick={() => setIsGrading(false)}>
                Voir toutes les recettes
              </Button>
            </motion.div>
          </motion.div>
        </CardContent>
      </Card>
    </motion.div>
  )

  const LoadingSkeleton = () => (
    <Card className="overflow-hidden">
      <AspectRatio ratio={16 / 10}>
        <Skeleton className="h-full w-full" />
      </AspectRatio>
      <CardHeader className="pb-2">
        <Skeleton className="h-5 w-3/4" />
        <Skeleton className="h-3 w-1/2 mt-2" />
      </CardHeader>
      <CardContent className="pb-2">
        <div className="flex gap-2">
          <Skeleton className="h-5 w-16 rounded-full" />
          <Skeleton className="h-5 w-20 rounded-full" />
        </div>
      </CardContent>
      <CardFooter className="pt-2">
        <Skeleton className="h-8 w-full" />
      </CardFooter>
    </Card>
  )

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
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
                <ChefHat className="h-6 w-6 text-primary" />
                {isGrading && !gradingComplete ? 'Note ces recettes' : 'Recommandations personnalisees'}
              </h1>
              <p className="text-sm text-muted-foreground mt-1">
                {isGrading && !gradingComplete
                  ? 'Dis-nous ce que tu aimes pour ameliorer tes recommandations'
                  : recommendations.length > 0
                    ? `${recommendations.length} recettes selectionnees selon tes preferences`
                    : 'Basees sur tes preferences et objectifs alimentaires'
                }
              </p>
            </div>
            {!isGrading && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleRefresh}
                disabled={refreshing || loading}
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
                Rafraichir
              </Button>
            )}
          </div>

          {/* Loading state - generating */}
          {(loading || refreshing) && status === 'generating' && (
            <Card className="border-dashed border-2">
              <CardContent className="flex flex-col items-center justify-center py-16">
                <div className="relative">
                  <Loader2 className="h-16 w-16 text-primary animate-spin" />
                  <ChefHat className="h-6 w-6 text-primary absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                </div>
                <p className="text-lg font-semibold mt-6">Generation des recommandations...</p>
                <p className="text-sm text-muted-foreground mt-2 text-center max-w-md">
                  On analyse tes preferences pour te proposer les meilleures recettes. Ca ne prendra que quelques secondes !
                </p>
              </CardContent>
            </Card>
          )}

          {/* Loading state - fetching */}
          {loading && status !== 'generating' && (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {[...Array(8)].map((_, i) => (
                <LoadingSkeleton key={i} />
              ))}
            </div>
          )}

          {/* Grading mode - Tinder style */}
          {!loading && !refreshing && recipesToGrade.length > 0 && isGrading && !gradingComplete && currentRecipeIndex < recipesToGrade.length && (
            <GradingSection />
          )}

          {/* Grading complete */}
          {!loading && !refreshing && gradingComplete && (
            <GradingComplete />
          )}

          {/* Recipes grid - after grading is complete */}
          {!loading && !refreshing && recommendations.length > 0 && !isGrading && (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {recommendations.map((recipe) => (
                <RecipeCard key={recipe.recipeid} recipe={recipe} />
              ))}
            </div>
          )}

          {/* Empty state */}
          {!loading && !refreshing && recommendations.length === 0 && status === 'not_found' && (
            <Card className="border-dashed border-2">
              <CardContent className="flex flex-col items-center justify-center py-16">
                <div className="rounded-full bg-muted p-4">
                  <ChefHat className="h-12 w-12 text-muted-foreground" />
                </div>
                <p className="text-lg font-semibold mt-6">Pas encore de recommandations</p>
                <p className="text-sm text-muted-foreground mt-2 text-center max-w-md">
                  Complete ton profil et tes preferences alimentaires pour recevoir des recommandations personnalisees
                </p>
                <Button className="mt-6" onClick={handleRefresh}>
                  Generer mes recommandations
                </Button>
              </CardContent>
            </Card>
          )}
        </div>
      </SidebarInset>

      {/* Recipe Detail Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        {selectedRecipe && <RecipeDetailDialog recipe={selectedRecipe} />}
      </Dialog>
    </SidebarProvider>
  )
}