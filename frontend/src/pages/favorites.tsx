import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
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
import { Heart, HeartOff, Trash2 } from 'lucide-react'
import { useState } from 'react'

export function Favorites() {
  const [favorites, setFavorites] = useState([
    {
      id: 1,
      name: 'Mediterranean Grilled Chicken',
      description: 'Tender grilled chicken with herbs, served with roasted vegetables and quinoa',
      tags: ['Protein-rich', 'Healthy', 'Mediterranean'],
      calories: 450,
      savedAt: '2025-01-15',
    },
    {
      id: 2,
      name: 'Salmon Poke Bowl',
      description: 'Fresh salmon with avocado, edamame, and brown rice in a soy-ginger dressing',
      tags: ['Omega-3', 'Fresh', 'Asian'],
      calories: 520,
      savedAt: '2025-01-18',
    },
  ])

  const removeFavorite = (id: number) => {
    setFavorites(favorites.filter((fav) => fav.id !== id))
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
                <BreadcrumbPage>Favorites</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </header>
        <div className="flex flex-1 flex-col gap-4 p-4">
          <div className="grid gap-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold tracking-tight">My Favorites</h1>
                <p className="text-muted-foreground">
                  {favorites.length} saved {favorites.length === 1 ? 'recipe' : 'recipes'}
                </p>
              </div>
              <Heart className="h-8 w-8 text-red-500 fill-red-500" />
            </div>

            {favorites.length > 0 ? (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {favorites.map((item) => (
                  <Card key={item.id} className="overflow-hidden">
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div className="space-y-1">
                          <CardTitle className="text-lg">{item.name}</CardTitle>
                          <CardDescription className="text-sm">
                            {item.calories} calories
                          </CardDescription>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => removeFavorite(item.id)}
                          className="text-red-500 hover:text-red-600 hover:bg-red-50"
                        >
                          <Trash2 className="h-5 w-5" />
                        </Button>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <p className="text-sm text-muted-foreground">{item.description}</p>
                      <div className="flex flex-wrap gap-2">
                        {item.tags.map((tag) => (
                          <Badge key={tag} variant="secondary">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                      <div className="flex items-center justify-between pt-2">
                        <span className="text-xs text-muted-foreground">
                          Saved {new Date(item.savedAt).toLocaleDateString()}
                        </span>
                        <Button className="w-auto">View Recipe</Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <Card>
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <HeartOff className="h-12 w-12 text-muted-foreground mb-4" />
                  <p className="text-lg font-medium">No favorites yet</p>
                  <p className="text-sm text-muted-foreground">
                    Start saving your favorite recipes to see them here
                  </p>
                  <Button className="mt-4">Browse Recommendations</Button>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}