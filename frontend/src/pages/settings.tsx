import { useAuth } from '@/hooks/useAuth'
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
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useEffect, useState } from 'react'
import { getUserPreferences, type UserPreferencesResponse, type MealType } from '@/lib/api/preferences'
import { Clock, Utensils, Flame, Beef, ShieldAlert } from 'lucide-react'

export function Settings() {
  const { user, profile } = useAuth()
  const [preferences, setPreferences] = useState<UserPreferencesResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchPreferences = async () => {
      if (!profile?.id) {
        setIsLoading(false)
        return
      }

      const { data, error } = await getUserPreferences(profile.id)
      if (!error && data) {
        setPreferences(data)
      }
      setIsLoading(false)
    }

    fetchPreferences()
  }, [profile?.id])

  const formatMealType = (mealType: string) => {
    return mealType.split('_').map(word =>
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ')
  }

  const formatGoal = (goal: string) => {
    return goal.charAt(0).toUpperCase() + goal.slice(1)
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
                <BreadcrumbPage>Settings</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </header>
        <div className="flex flex-1 flex-col gap-4 p-4">
          <div className="grid gap-6 max-w-2xl">
            <Card>
              <CardHeader>
                <CardTitle>Account Information</CardTitle>
                <CardDescription>Manage your account details and preferences</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input id="email" type="email" value={user?.email || ''} disabled />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="firstName">First Name</Label>
                    <Input id="firstName" value={profile?.first_name || ''} disabled />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lastName">Last Name</Label>
                    <Input id="lastName" value={profile?.last_name || ''} disabled />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Preferences</CardTitle>
                <CardDescription>Your food recommendation preferences</CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <p className="text-sm text-muted-foreground">Loading preferences...</p>
                ) : !preferences ? (
                  <p className="text-sm text-muted-foreground">
                    No preferences set yet. Complete the signup flow to set your preferences.
                  </p>
                ) : (
                  <div className="space-y-6">
                    {/* Meal Types */}
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <Utensils className="h-4 w-4 text-muted-foreground" />
                        <Label>Preferred Meal Types</Label>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {preferences.meal_types.map((mealType: MealType) => (
                          <Badge key={mealType} variant="secondary">
                            {formatMealType(mealType)}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    {/* Max Cooking Time */}
                    {preferences.max_total_time && (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Clock className="h-4 w-4 text-muted-foreground" />
                          <Label>Maximum Cooking Time</Label>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {preferences.max_total_time} minutes
                        </p>
                      </div>
                    )}

                    {/* Calorie Goal */}
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <Flame className="h-4 w-4 text-muted-foreground" />
                        <Label>Calorie Goal</Label>
                      </div>
                      <Badge variant="outline">{formatGoal(preferences.calorie_goal)}</Badge>
                    </div>

                    {/* Protein Goal */}
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <Beef className="h-4 w-4 text-muted-foreground" />
                        <Label>Protein Goal</Label>
                      </div>
                      <Badge variant="outline">{formatGoal(preferences.protein_goal)}</Badge>
                    </div>

                    {/* Dietary Restrictions */}
                    {preferences.dietary_restrictions && preferences.dietary_restrictions.length > 0 && (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <ShieldAlert className="h-4 w-4 text-muted-foreground" />
                          <Label>Dietary Restrictions</Label>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {preferences.dietary_restrictions.map((restriction: string) => (
                            <Badge key={restriction} variant="destructive">
                              {restriction}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Danger Zone</CardTitle>
                <CardDescription>Irreversible actions</CardDescription>
              </CardHeader>
              <CardContent>
                <Button variant="destructive" disabled>
                  Delete Account
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}