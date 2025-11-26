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
import { Heart, ChefHat, Clock, Star, Loader2, Flame, Drumstick, Leaf, UtensilsCrossed, RefreshCw } from 'lucide-react'
import { getUserRecommendations, triggerRecommendations, type Recipe, type RecommendationsResponse } from '@/lib/api/recommendations'
import { getUserProfile } from '@/lib/api/auth'

export function Recommendations() {
  const { user } = useAuth()
  const [recommendations, setRecommendations] = useState<Recipe[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [status, setStatus] = useState<RecommendationsResponse['status']>('generating')

  const fetchRecommendations = async () => {
    if (!user) return

    try {
      const userProfile = await getUserProfile(user.id)
      if (!userProfile) return

      const { data } = await getUserRecommendations(userProfile.id)

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

  const RecipeCard = ({ recipe }: { recipe: Recipe }) => {
    const imageUrl = recipe.images?.[0] || null
    const [liked, setLiked] = useState(false)

    return (
      <Card className="group overflow-hidden transition-all hover:shadow-lg">
        <div className="relative">
          <AspectRatio ratio={16 / 10}>
            {imageUrl ? (
              <img
                src={imageUrl}
                alt={recipe.name}
                className="h-full w-full object-cover transition-transform group-hover:scale-105"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = 'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=800&auto=format&fit=crop'
                }}
              />
            ) : (
              <div className="flex h-full w-full items-center justify-center bg-muted">
                <UtensilsCrossed className="h-12 w-12 text-muted-foreground/50" />
              </div>
            )}
          </AspectRatio>
          <Button
            variant="secondary"
            size="icon"
            className="absolute right-2 top-2 h-8 w-8 rounded-full opacity-0 transition-opacity group-hover:opacity-100"
            onClick={() => setLiked(!liked)}
          >
            <Heart className={`h-4 w-4 ${liked ? 'fill-red-500 text-red-500' : ''}`} />
          </Button>
          {recipe.aggregatedrating && recipe.aggregatedrating >= 4.5 && (
            <Badge className="absolute left-2 top-2 bg-yellow-500 text-yellow-950">
              <Star className="mr-1 h-3 w-3 fill-current" />
              Top note
            </Badge>
          )}
        </div>

        <CardHeader className="pb-2">
          <CardTitle className="line-clamp-2 text-base">{recipe.name}</CardTitle>
          <CardDescription className="flex items-center gap-3 text-xs">
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
                ({recipe.reviewcount} avis)
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
          <Button variant="default" className="w-full" size="sm">
            Voir la recette
          </Button>
        </CardFooter>
      </Card>
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
    </SidebarProvider>
  )
}