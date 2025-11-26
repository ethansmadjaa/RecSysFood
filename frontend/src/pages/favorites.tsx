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
import { Heart, Clock, Star, Flame, Drumstick, Leaf, Users, Calendar, AlertTriangle, Utensils, Timer, HeartOff } from 'lucide-react'
import { type Recipe } from '@/lib/api/recommendations'
import { getUserProfile } from '@/lib/api/auth'
import { getUserFavorites, removeFavorite } from '@/lib/api/favorites'
import { toast } from 'sonner'

export function Favorites() {
  const { user } = useAuth()
  const [favorites, setFavorites] = useState<Recipe[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedRecipe, setSelectedRecipe] = useState<Recipe | null>(null)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [userProfileId, setUserProfileId] = useState<string | null>(null)

  useEffect(() => {
    const fetchFavorites = async () => {
      if (!user) return

      try {
        const userProfile = await getUserProfile(user.id)
        if (!userProfile) {
          setLoading(false)
          return
        }

        setUserProfileId(userProfile.id)

        const { data, error } = await getUserFavorites(userProfile.id)

        if (error) {
          console.error('Error fetching favorites:', error)
          toast.error('Erreur lors du chargement des favoris')
        }

        if (data) {
          setFavorites(data)
        }
      } catch (error) {
        console.error('Error fetching favorites:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchFavorites()
  }, [user])

  const handleRemoveFavorite = async (recipeId: number) => {
    if (!userProfileId) return

    // Optimistic update
    setFavorites(prev => prev.filter(r => r.recipeid !== recipeId))

    const { success, error } = await removeFavorite(userProfileId, recipeId)

    if (!success) {
      // Revert on error - refetch
      const { data } = await getUserFavorites(userProfileId)
      if (data) setFavorites(data)
      toast.error(error || 'Erreur lors de la suppression du favori')
    } else {
      toast.success('Retire des favoris')
    }
  }

  // Curated list of high-quality food images from Unsplash
  const foodImageIds = [
    'photo-1546069901-ba9599a7e63c',
    'photo-1567620905732-2d1ec7ab7445',
    'photo-1565299624946-b28f40a0ae38',
    'photo-1540189549336-e6e99c3679fe',
    'photo-1565958011703-44f9829ba187',
    'photo-1482049016gy-7db4v34g7e',
    'photo-1504674900247-0877df9cc836',
    'photo-1512621776951-a57141f2eefd',
    'photo-1473093295043-cdd812d0e601',
    'photo-1476224203421-9ac39bcb3327',
    'photo-1484723091739-30a097e8f929',
    'photo-1432139555190-58524dae6a55',
    'photo-1529042410759-befb1204b468',
    'photo-1414235077428-338989a2e8c0',
    'photo-1490645935967-10de6ba17061',
    'photo-1498837167922-ddd27525d352',
    'photo-1455619452474-d2be8b1e70cd',
    'photo-1476718406336-bb5a9690ee2a',
    'photo-1499028344343-cd173ffc68a9',
    'photo-1547592180-85f173990554',
    'photo-1493770348161-369560ae357d',
    'photo-1506354666786-959d6d497f1a',
    'photo-1551183053-bf91a1d81141',
    'photo-1559847844-5315695dadae',
    'photo-1574484284002-952d92456975',
    'photo-1585937421612-70a008356fbe',
    'photo-1563379926898-05f4575a45d8',
    'photo-1569718212165-3a8278d5f624',
    'photo-1604908176997-125f25cc6f3d',
    'photo-1594007654729-407eedc4be65',
  ]

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
            className="absolute right-2 top-2 h-8 w-8 rounded-full opacity-100"
            onClick={(e) => {
              e.stopPropagation()
              handleRemoveFavorite(recipe.recipeid)
            }}
          >
            <Heart className="h-4 w-4 fill-red-500 text-red-500" />
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

    const ingredients = recipe.recipeingredientparts?.map((part, index) => ({
      name: part,
      quantity: recipe.recipeingredientquantities?.[index] || ''
    })) || []

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
            variant="default"
            size="sm"
            onClick={() => handleRemoveFavorite(recipe.recipeid)}
            className="shrink-0"
          >
            <Heart className="h-4 w-4 mr-2 fill-current" />
            Retirer des favoris
          </Button>
        </DialogHeader>

        <ScrollArea className="flex-1 pr-4">
          <div className="space-y-6">
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

            <div className="space-y-4">
              {recipe.description && (
                <p className="text-sm text-muted-foreground">{recipe.description}</p>
              )}

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
                <BreadcrumbPage>Favoris</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </header>

        <div className="flex flex-1 flex-col gap-6 p-4 md:p-6">
          <div>
            <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
              <Heart className="h-6 w-6 text-red-500" />
              Mes favoris
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              {favorites.length > 0
                ? `${favorites.length} recette${favorites.length > 1 ? 's' : ''} sauvegardee${favorites.length > 1 ? 's' : ''}`
                : 'Tes recettes preferees apparaitront ici'
              }
            </p>
          </div>

          {/* Loading state */}
          {loading && (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {[...Array(8)].map((_, i) => (
                <LoadingSkeleton key={i} />
              ))}
            </div>
          )}

          {/* Favorites grid */}
          {!loading && favorites.length > 0 && (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {favorites.map((recipe) => (
                <RecipeCard key={recipe.recipeid} recipe={recipe} />
              ))}
            </div>
          )}

          {/* Empty state */}
          {!loading && favorites.length === 0 && (
            <Card className="border-dashed border-2">
              <CardContent className="flex flex-col items-center justify-center py-16">
                <div className="rounded-full bg-muted p-4">
                  <HeartOff className="h-12 w-12 text-muted-foreground" />
                </div>
                <p className="text-lg font-semibold mt-6">Aucun favori pour l'instant</p>
                <p className="text-sm text-muted-foreground mt-2 text-center max-w-md">
                  Explore les recommandations et clique sur le coeur pour sauvegarder tes recettes preferees
                </p>
                <Button className="mt-6" onClick={() => window.location.href = '/recommendations'}>
                  Voir les recommandations
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
