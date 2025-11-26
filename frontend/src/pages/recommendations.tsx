import { useEffect, useState } from 'react'
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
import { Heart, ChefHat, Clock, Star, Loader2, Flame, Drumstick, Leaf, RefreshCw, Users, Calendar, AlertTriangle, Utensils, Timer } from 'lucide-react'
import { getUserRecommendations, triggerRecommendations, type Recipe, type RecommendationsResponse } from '@/lib/api/recommendations'
import { getUserProfile } from '@/lib/api/auth'

export function Recommendations() {
  const { user } = useAuth()
  const [recommendations, setRecommendations] = useState<Recipe[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [status, setStatus] = useState<RecommendationsResponse['status']>('generating')
  const [selectedRecipe, setSelectedRecipe] = useState<Recipe | null>(null)
  const [dialogOpen, setDialogOpen] = useState(false)

  const fetchRecommendations = async () => {
    if (!user) return

    try {
      const userProfile = await getUserProfile(user.id)
      if (!userProfile) return

      const { data } = await getUserRecommendations(userProfile.id)

      console.log('data', data)

      if (data) {
        setStatus(data.status)
        if (data.status === 'ready' && data.recipes.length > 0) {
          setRecommendations(data.recipes)
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
    'photo-1482049016gy-7db4v34g7e', // pasta
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
    const [liked, setLiked] = useState(false)

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
            className="absolute right-2 top-2 h-8 w-8 rounded-full opacity-0 transition-opacity group-hover:opacity-100"
            onClick={(e) => {
              e.stopPropagation()
              setLiked(!liked)
            }}
          >
            <Heart className={`h-4 w-4 ${liked ? 'fill-red-500 text-red-500' : ''}`} />
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
        <DialogHeader>
          <DialogTitle className="text-xl">{recipe.name}</DialogTitle>
          {recipe.authorname && (
            <p className="text-sm text-muted-foreground">Par {recipe.authorname}</p>
          )}
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
                {recipe.recipeinstructions && recipe.recipeinstructions.length > 0 ? (
                  <ol className="space-y-3">
                    {recipe.recipeinstructions.map((instruction, index) => (
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
                )}
              </TabsContent>

              <TabsContent value="nutrition" className="mt-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {recipe.calories && (
                    <div className="p-3 bg-muted rounded-lg text-center">
                      <Flame className="h-5 w-5 mx-auto text-orange-500" />
                      <p className="text-lg font-bold mt-1">{Math.round(recipe.calories)}</p>
                      <p className="text-xs text-muted-foreground">Calories</p>
                    </div>
                  )}
                  {recipe.proteincontent && (
                    <div className="p-3 bg-muted rounded-lg text-center">
                      <Drumstick className="h-5 w-5 mx-auto text-red-500" />
                      <p className="text-lg font-bold mt-1">{Math.round(recipe.proteincontent)}g</p>
                      <p className="text-xs text-muted-foreground">Proteines</p>
                    </div>
                  )}
                  {recipe.carbohydratecontent && (
                    <div className="p-3 bg-muted rounded-lg text-center">
                      <div className="h-5 w-5 mx-auto text-yellow-500 font-bold text-sm">C</div>
                      <p className="text-lg font-bold mt-1">{Math.round(recipe.carbohydratecontent)}g</p>
                      <p className="text-xs text-muted-foreground">Glucides</p>
                    </div>
                  )}
                  {recipe.fatcontent && (
                    <div className="p-3 bg-muted rounded-lg text-center">
                      <div className="h-5 w-5 mx-auto text-blue-500 font-bold text-sm">F</div>
                      <p className="text-lg font-bold mt-1">{Math.round(recipe.fatcontent)}g</p>
                      <p className="text-xs text-muted-foreground">Lipides</p>
                    </div>
                  )}
                  {recipe.fibercontent && (
                    <div className="p-3 bg-muted rounded-lg text-center">
                      <Leaf className="h-5 w-5 mx-auto text-green-500" />
                      <p className="text-lg font-bold mt-1">{Math.round(recipe.fibercontent)}g</p>
                      <p className="text-xs text-muted-foreground">Fibres</p>
                    </div>
                  )}
                  {recipe.sugarcontent && (
                    <div className="p-3 bg-muted rounded-lg text-center">
                      <div className="h-5 w-5 mx-auto text-pink-500 font-bold text-sm">S</div>
                      <p className="text-lg font-bold mt-1">{Math.round(recipe.sugarcontent)}g</p>
                      <p className="text-xs text-muted-foreground">Sucres</p>
                    </div>
                  )}
                  {recipe.sodiumcontent && (
                    <div className="p-3 bg-muted rounded-lg text-center">
                      <div className="h-5 w-5 mx-auto text-gray-500 font-bold text-sm">Na</div>
                      <p className="text-lg font-bold mt-1">{Math.round(recipe.sodiumcontent)}mg</p>
                      <p className="text-xs text-muted-foreground">Sodium</p>
                    </div>
                  )}
                  {recipe.cholesterolcontent && (
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
                Recommandations personnalisees
              </h1>
              <p className="text-sm text-muted-foreground mt-1">
                {recommendations.length > 0
                  ? `${recommendations.length} recettes selectionnees selon tes preferences`
                  : 'Basees sur tes preferences et objectifs alimentaires'
                }
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              disabled={refreshing || loading}
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
              Rafraichir
            </Button>
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

          {/* Recipes grid */}
          {!loading && !refreshing && recommendations.length > 0 && (
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