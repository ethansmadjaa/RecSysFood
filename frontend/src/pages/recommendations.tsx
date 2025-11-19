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
import { Heart, ChefHat } from 'lucide-react'

export function Recommendations() {

  // Mock recommendations data
  const recommendations = [
    {
      id: 1,
      name: 'Mediterranean Grilled Chicken',
      description: 'Tender grilled chicken with herbs, served with roasted vegetables and quinoa',
      tags: ['Protein-rich', 'Healthy', 'Mediterranean'],
      calories: 450,
    },
    {
      id: 2,
      name: 'Salmon Poke Bowl',
      description: 'Fresh salmon with avocado, edamame, and brown rice in a soy-ginger dressing',
      tags: ['Omega-3', 'Fresh', 'Asian'],
      calories: 520,
    },
    {
      id: 3,
      name: 'Vegetarian Buddha Bowl',
      description: 'Colorful mix of roasted chickpeas, sweet potato, kale, and tahini dressing',
      tags: ['Vegan', 'High-fiber', 'Nutrient-dense'],
      calories: 380,
    },
  ]

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
                <BreadcrumbPage>Recommendations</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </header>
        <div className="flex flex-1 flex-col gap-4 p-4">
          <div className="grid gap-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold tracking-tight">
                  Personalized Recommendations
                </h1>
                <p className="text-muted-foreground">
                  Based on your preferences and dietary goals
                </p>
              </div>
              <ChefHat className="h-8 w-8 text-muted-foreground" />
            </div>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {recommendations.map((item) => (
                <Card key={item.id} className="overflow-hidden">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="space-y-1">
                        <CardTitle className="text-lg">{item.name}</CardTitle>
                        <CardDescription className="text-sm">
                          {item.calories} calories
                        </CardDescription>
                      </div>
                      <Button variant="ghost" size="icon">
                        <Heart className="h-5 w-5" />
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
                    <Button className="w-full">View Recipe</Button>
                  </CardContent>
                </Card>
              ))}
            </div>

            {recommendations.length === 0 && (
              <Card>
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <ChefHat className="h-12 w-12 text-muted-foreground mb-4" />
                  <p className="text-lg font-medium">No recommendations yet</p>
                  <p className="text-sm text-muted-foreground">
                    Complete your profile to get personalized food recommendations
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}