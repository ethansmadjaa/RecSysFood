import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { AspectRatio } from '@/components/ui/aspect-ratio'
import { Heart, Clock, Star, Flame, Drumstick, Leaf, Users, Sparkles } from 'lucide-react'
import type { Recipe } from '@/lib/api/recommendations'
import { getRecipeImageUrl, DEFAULT_FOOD_IMAGE } from './recipe-images'

interface RecipeCardProps {
  recipe: Recipe
  isFavorite: boolean
  onToggleFavorite: (recipeId: number) => void
  onViewDetails: (recipe: Recipe) => void
}

export function RecipeCard({
  recipe,
  isFavorite,
  onToggleFavorite,
  onViewDetails,
}: RecipeCardProps) {
  const imageUrl = getRecipeImageUrl(recipe.images, recipe.recipeid)

  return (
    <Card className="group overflow-hidden transition-all hover:shadow-lg">
      <div className="relative">
        <AspectRatio ratio={16 / 10}>
          <img
            src={imageUrl}
            alt={recipe.name}
            className="h-full w-full object-cover transition-transform group-hover:scale-105"
            onError={(e) => {
              (e.target as HTMLImageElement).src = DEFAULT_FOOD_IMAGE
            }}
          />
        </AspectRatio>
        <Button
          variant="secondary"
          size="icon"
          className={`absolute right-2 top-2 h-8 w-8 rounded-full transition-opacity ${isFavorite ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}
          onClick={(e) => {
            e.stopPropagation()
            onToggleFavorite(recipe.recipeid)
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
        {recipe.reason && (
          <p className="flex items-center gap-1.5 text-xs text-purple-600 mt-1">
            <Sparkles className="h-3 w-3 shrink-0" />
            <span className="line-clamp-2">{recipe.reason}</span>
          </p>
        )}
        {recipe.description && !recipe.reason && (
          <p className="text-xs text-muted-foreground line-clamp-2 mt-1">
            {recipe.description}
          </p>
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
            <Badge
              variant="outline"
              className="text-xs bg-green-50 text-green-700 border-green-200"
            >
              <Leaf className="mr-1 h-3 w-3" />
              Vegan
            </Badge>
          )}
          {recipe.is_vegetarian && !recipe.is_vegan && (
            <Badge
              variant="outline"
              className="text-xs bg-green-50 text-green-700 border-green-200"
            >
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
            <Badge
              variant="outline"
              className="text-xs bg-blue-50 text-blue-700 border-blue-200"
            >
              <Drumstick className="mr-1 h-3 w-3" />
              Riche en proteines
            </Badge>
          )}
        </div>
      </CardContent>

      <CardFooter className="pt-2">
        <Button
          variant="default"
          className="w-full"
          size="sm"
          onClick={() => onViewDetails(recipe)}
        >
          Voir la recette
        </Button>
      </CardFooter>
    </Card>
  )
}
