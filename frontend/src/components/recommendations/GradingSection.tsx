import { motion, AnimatePresence } from 'framer-motion'
import type { Recipe } from '@/lib/api/recommendations'
import { GradingCard } from './GradingCard'

interface GradingSectionProps {
  recipesToGrade: Recipe[]
  currentRecipeIndex: number
  onRate: (rating: 0 | 1 | 2) => void
  onViewDetails: (recipe: Recipe) => void
  submittingRating: boolean
  swipeDirection: 'left' | 'right' | 'down' | null
}

export function GradingSection({
  recipesToGrade,
  currentRecipeIndex,
  onRate,
  onViewDetails,
  submittingRating,
  swipeDirection,
}: GradingSectionProps) {
  const currentRecipe = recipesToGrade[currentRecipeIndex]
  if (!currentRecipe) return null

  return (
    <div className="flex flex-col items-center justify-center w-full max-w-lg mx-auto">
      {/* Progress indicator */}
      <div className="w-full mb-4">
        <div className="flex justify-between text-sm text-muted-foreground mb-2">
          <span>
            Recette {currentRecipeIndex + 1} sur {recipesToGrade.length}
          </span>
          <span>
            {Math.round((currentRecipeIndex / recipesToGrade.length) * 100)}%
          </span>
        </div>
        <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
          <motion.div
            className="h-2 rounded-full"
            style={{ backgroundColor: '#3568F6' }}
            initial={{ width: 0 }}
            animate={{
              width: `${(currentRecipeIndex / recipesToGrade.length) * 100}%`,
            }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
          />
        </div>
      </div>

      {/* Card stack with AnimatePresence */}
      <div className="relative w-full h-[600px]">
        <AnimatePresence mode="wait">
          <GradingCard
            key={currentRecipe.recipeid}
            recipe={currentRecipe}
            onRate={onRate}
            onViewDetails={onViewDetails}
            submittingRating={submittingRating}
            swipeDirection={swipeDirection}
          />
        </AnimatePresence>
      </div>
    </div>
  )
}
