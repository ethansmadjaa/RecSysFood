import { Card, CardContent, CardHeader, CardFooter } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { AspectRatio } from '@/components/ui/aspect-ratio'

export function RecipeCardSkeleton() {
  return (
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
}
