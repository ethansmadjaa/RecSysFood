import { motion, useMotionValue, useTransform, animate } from 'framer-motion'
import type { PanInfo } from 'framer-motion'
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { AspectRatio } from '@/components/ui/aspect-ratio'
import {
  Clock,
  Star,
  Flame,
  Drumstick,
  Leaf,
  Users,
  Utensils,
  ThumbsDown,
  ThumbsUp,
  Meh,
} from 'lucide-react'
import type { Recipe } from '@/lib/api/recommendations'
import { getRecipeImageUrl, DEFAULT_FOOD_IMAGE } from './recipe-images'

interface GradingCardProps {
  recipe: Recipe
  onRate: (rating: 0 | 1 | 2) => void
  onViewDetails: (recipe: Recipe) => void
  submittingRating: boolean
  swipeDirection: 'left' | 'right' | 'down' | null
}

export function GradingCard({
  recipe,
  onRate,
  onViewDetails,
  submittingRating,
  swipeDirection,
}: GradingCardProps) {
  const imageUrl = getRecipeImageUrl(recipe.images, recipe.recipeid)

  // Motion values for drag
  const x = useMotionValue(0)
  const y = useMotionValue(0)

  // Transform values based on drag position
  const rotate = useTransform(x, [-300, 0, 300], [-25, 0, 25])
  const likeOpacity = useTransform(x, [0, 100, 200], [0, 0.5, 1])
  const dislikeOpacity = useTransform(x, [-200, -100, 0], [1, 0.5, 0])
  const mehOpacity = useTransform(y, [0, 100, 200], [0, 0.5, 1])

  // Scale effect while dragging
  const scale = useTransform(x, [-300, -150, 0, 150, 300], [0.95, 0.98, 1, 0.98, 0.95])

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
        transition: { duration: 0.3, ease: 'easeOut' },
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
              src={imageUrl}
              alt={recipe.name}
              className="h-full w-full object-cover"
              draggable={false}
              onError={(e) => {
                (e.target as HTMLImageElement).src = DEFAULT_FOOD_IMAGE
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
            <p className="text-sm text-muted-foreground line-clamp-2 mt-1">
              {recipe.description}
            </p>
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
            {recipe.calories && (
              <Badge variant="outline">
                <Flame className="mr-1 h-3 w-3" />
                {Math.round(recipe.calories)} cal
              </Badge>
            )}
            {recipe.proteincontent && recipe.proteincontent > 20 && (
              <Badge
                variant="outline"
                className="bg-blue-50 text-blue-700 border-blue-200"
              >
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
              onViewDetails(recipe)
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
                borderColor: 'rgb(239, 68, 68)',
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
                borderColor: 'rgb(107, 114, 128)',
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
                borderColor: 'rgb(34, 197, 94)',
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
