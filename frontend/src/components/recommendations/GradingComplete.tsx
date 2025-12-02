import { motion } from 'framer-motion'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ThumbsUp } from 'lucide-react'

interface GradingCompleteProps {
  onViewRecipes: () => void
}

export function GradingComplete({ onViewRecipes }: GradingCompleteProps) {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 200, damping: 20 }}
    >
      <Card className="w-full max-w-lg mx-auto overflow-hidden">
        <CardContent className="flex flex-col items-center justify-center py-12">
          <motion.div
            className="rounded-full p-5 mb-6"
            style={{ backgroundColor: '#dcfce7' }}
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ type: 'spring', stiffness: 200, damping: 15, delay: 0.2 }}
          >
            <motion.div
              animate={{
                scale: [1, 1.2, 1],
              }}
              transition={{ repeat: 2, duration: 0.4, delay: 0.5 }}
            >
              <ThumbsUp className="h-14 w-14 text-green-600" />
            </motion.div>
          </motion.div>
          <motion.h2
            className="text-2xl font-bold text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            Merci pour tes notes !
          </motion.h2>
          <motion.p
            className="text-muted-foreground text-center mt-2 max-w-md"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            Tes preferences ont ete enregistrees. Nous allons ameliorer tes
            recommandations.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button className="mt-6" size="lg" onClick={onViewRecipes}>
                Voir toutes les recettes
              </Button>
            </motion.div>
          </motion.div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
