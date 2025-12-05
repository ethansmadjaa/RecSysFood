import { DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Heart,
  Clock,
  Star,
  Flame,
  Drumstick,
  Leaf,
  Users,
  Calendar,
  AlertTriangle,
  Utensils,
  Timer,
} from 'lucide-react'
import type { Recipe } from '@/lib/api/recommendations'
import { getRecipeImageUrl, DEFAULT_FOOD_IMAGE } from './recipe-images'

interface RecipeDetailDialogProps {
  recipe: Recipe
  isFavorite: boolean
  onToggleFavorite: (recipeId: number) => void
}

export function RecipeDetailDialog({
  recipe,
  isFavorite,
  onToggleFavorite,
}: RecipeDetailDialogProps) {
  const imageUrl = getRecipeImageUrl(recipe.images, recipe.recipeid)

  // Combine ingredients with quantities
  const ingredients =
    recipe.recipeingredientparts?.map((part, index) => ({
      name: part,
      quantity: recipe.recipeingredientquantities?.[index] || '',
    })) || []

  // Allergens/dietary warnings
  const warnings: string[] = []
  if (recipe.contains_nuts) warnings.push('Noix')
  if (recipe.contains_dairy) warnings.push('Produits laitiers')
  if (recipe.contains_egg) warnings.push('Oeufs')
  if (recipe.contains_fish) warnings.push('Poisson')
  if (recipe.contains_soy) warnings.push('Soja')
  if (recipe.contains_gluten) warnings.push('Gluten')
  if (recipe.contains_pork) warnings.push('Porc')
  if (recipe.contains_alcohol) warnings.push('Alcool')

  // Parse instructions
  const getInstructions = (): string[] => {
    if (!recipe.recipeinstructions) return []
    if (Array.isArray(recipe.recipeinstructions)) {
      return recipe.recipeinstructions
    }
    if (typeof recipe.recipeinstructions === 'string') {
      const str = recipe.recipeinstructions.trim()
      // Check if it's a Python-style array string: ['item1', 'item2']
      if (str.startsWith('[') && str.endsWith(']')) {
        // Extract content between brackets and split by "', '"
        const content = str.slice(2, -2) // Remove [' and ']
        return content.split("', '").map((s: string) => s.trim())
      }
      return [str]
    }
    return []
  }

  const instructions = getInstructions()

  return (
    <DialogContent className="max-w-4xl  overflow-hidden flex flex-col p-0">
      <DialogHeader className="flex flex-row items-start justify-between gap-4 p-6 pb-0">
        <div className="flex-1">
          <DialogTitle className="text-xl">{recipe.name}</DialogTitle>
          {recipe.authorname && (
            <p className="text-sm text-muted-foreground">Par {recipe.authorname}</p>
          )}
        </div>
        <Button
          variant={isFavorite ? 'default' : 'outline'}
          size="sm"
          onClick={() => onToggleFavorite(recipe.recipeid)}
          className="shrink-0"
        >
          <Heart className={`h-4 w-4 mr-2 ${isFavorite ? 'fill-current' : ''}`} />
          {isFavorite ? 'Favori' : 'Ajouter aux favoris'}
        </Button>
      </DialogHeader>

      <ScrollArea className="flex-1 min-h-0">
        <div className="space-y-6 p-6 pt-4">
          {/* Image */}
          <div className="rounded-lg overflow-hidden bg-muted">
            <img
              src={imageUrl}
              alt={recipe.name}
              className="w-full h-auto max-h-[250px] object-cover"
              onError={(e) => {
                (e.target as HTMLImageElement).src = DEFAULT_FOOD_IMAGE
              }}
            />
          </div>

          {/* Main info */}
          <div className="space-y-4">
            {recipe.description && (
              <p className="text-sm text-muted-foreground">{recipe.description}</p>
            )}

            {/* Time and servings */}
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
                    <p className="text-sm font-medium">
                      {recipe.recipeservings} {recipe.recipeyield || 'portions'}
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Rating */}
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
                <span className="text-sm font-medium">
                  {recipe.aggregatedrating.toFixed(1)}
                </span>
                {recipe.reviewcount && (
                  <span className="text-sm text-muted-foreground">
                    ({Math.round(recipe.reviewcount)} avis)
                  </span>
                )}
              </div>
            )}

            {/* Tags */}
            <div className="flex flex-wrap gap-1.5">
              {recipe.recipecategory && (
                <Badge variant="secondary">{recipe.recipecategory}</Badge>
              )}
              {recipe.is_vegan && (
                <Badge
                  variant="outline"
                  className="bg-green-50 text-green-700 border-green-200"
                >
                  <Leaf className="mr-1 h-3 w-3" />
                  Vegan
                </Badge>
              )}
              {recipe.is_vegetarian && !recipe.is_vegan && (
                <Badge
                  variant="outline"
                  className="bg-green-50 text-green-700 border-green-200"
                >
                  <Leaf className="mr-1 h-3 w-3" />
                  Vegetarien
                </Badge>
              )}
              {recipe.is_breakfast_brunch && (
                <Badge variant="outline">Petit-dejeuner</Badge>
              )}
              {recipe.is_dessert && <Badge variant="outline">Dessert</Badge>}
            </div>

            {/* Allergen warnings */}
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

          {/* Tabs for ingredients, instructions, nutrition */}
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
                    <li
                      key={index}
                      className="flex items-center gap-2 p-2 bg-muted rounded-lg"
                    >
                      <span className="font-medium text-sm">{ingredient.quantity}</span>
                      <span className="text-sm">{ingredient.name}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">
                  Ingredients non disponibles
                </p>
              )}
            </TabsContent>

            <TabsContent value="instructions" className="mt-4">
              {instructions.length > 0 ? (
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
                <p className="text-sm text-muted-foreground">
                  Instructions non disponibles
                </p>
              )}
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
                    <p className="text-lg font-bold mt-1">
                      {Math.round(recipe.proteincontent)}g
                    </p>
                    <p className="text-xs text-muted-foreground">Proteines</p>
                  </div>
                )}
                {recipe.carbohydratecontent != null && recipe.carbohydratecontent > 0 && (
                  <div className="p-3 bg-muted rounded-lg text-center">
                    <div className="h-5 w-5 mx-auto text-yellow-500 font-bold text-sm">
                      C
                    </div>
                    <p className="text-lg font-bold mt-1">
                      {Math.round(recipe.carbohydratecontent)}g
                    </p>
                    <p className="text-xs text-muted-foreground">Glucides</p>
                  </div>
                )}
                {recipe.fatcontent != null && recipe.fatcontent > 0 && (
                  <div className="p-3 bg-muted rounded-lg text-center">
                    <div className="h-5 w-5 mx-auto text-blue-500 font-bold text-sm">F</div>
                    <p className="text-lg font-bold mt-1">
                      {Math.round(recipe.fatcontent)}g
                    </p>
                    <p className="text-xs text-muted-foreground">Lipides</p>
                  </div>
                )}
                {recipe.fibercontent != null && recipe.fibercontent > 0 && (
                  <div className="p-3 bg-muted rounded-lg text-center">
                    <Leaf className="h-5 w-5 mx-auto text-green-500" />
                    <p className="text-lg font-bold mt-1">
                      {Math.round(recipe.fibercontent)}g
                    </p>
                    <p className="text-xs text-muted-foreground">Fibres</p>
                  </div>
                )}
                {recipe.sugarcontent != null && recipe.sugarcontent > 0 && (
                  <div className="p-3 bg-muted rounded-lg text-center">
                    <div className="h-5 w-5 mx-auto text-pink-500 font-bold text-sm">S</div>
                    <p className="text-lg font-bold mt-1">
                      {Math.round(recipe.sugarcontent)}g
                    </p>
                    <p className="text-xs text-muted-foreground">Sucres</p>
                  </div>
                )}
                {recipe.sodiumcontent != null && recipe.sodiumcontent > 0 && (
                  <div className="p-3 bg-muted rounded-lg text-center">
                    <div className="h-5 w-5 mx-auto text-gray-500 font-bold text-sm">
                      Na
                    </div>
                    <p className="text-lg font-bold mt-1">
                      {Math.round(recipe.sodiumcontent)}mg
                    </p>
                    <p className="text-xs text-muted-foreground">Sodium</p>
                  </div>
                )}
                {recipe.cholesterolcontent != null && recipe.cholesterolcontent > 0 && (
                  <div className="p-3 bg-muted rounded-lg text-center">
                    <div className="h-5 w-5 mx-auto text-purple-500 font-bold text-sm">
                      Ch
                    </div>
                    <p className="text-lg font-bold mt-1">
                      {Math.round(recipe.cholesterolcontent)}mg
                    </p>
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

          {/* Publication date */}
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
