// Curated list of high-quality food images from Unsplash
const foodImageIds = [
  'photo-1546069901-ba9599a7e63c', // colorful salad bowl
  'photo-1567620905732-2d1ec7ab7445', // pancakes
  'photo-1565299624946-b28f40a0ae38', // pizza
  'photo-1540189549336-e6e99c3679fe', // food platter
  'photo-1565958011703-44f9829ba187', // dessert
  'photo-1621996346565-e3dbc646d9a9', // pasta
  'photo-1504674900247-0877df9cc836', // grilled food
  'photo-1512621776951-a57141f2eefd', // healthy bowl
  'photo-1473093295043-cdd812d0e601', // pasta dish
  'photo-1476224203421-9ac39bcb3327', // breakfast
  'photo-1484723091739-30a097e8f929', // french toast
  'photo-1432139555190-58524dae6a55', // salad
  'photo-1529042410759-befb1204b468', // ramen
  'photo-1414235077428-338989a2e8c0', // fine dining
  'photo-1490645935967-10de6ba17061', // food spread
  'photo-1498837167922-ddd27525d352', // vegetables
  'photo-1455619452474-d2be8b1e70cd', // asian food
  'photo-1476718406336-bb5a9690ee2a', // burger
  'photo-1499028344343-cd173ffc68a9', // tacos
  'photo-1547592180-85f173990554', // breakfast bowl
  'photo-1493770348161-369560ae357d', // healthy food
  'photo-1506354666786-959d6d497f1a', // smoothie bowl
  'photo-1551183053-bf91a1d81141', // sushi
  'photo-1559847844-5315695dadae', // curry
  'photo-1574484284002-952d92456975', // indian food
  'photo-1585937421612-70a008356fbe', // mediterranean
  'photo-1563379926898-05f4575a45d8', // pasta carbonara
  'photo-1569718212165-3a8278d5f624', // fried rice
  'photo-1604908176997-125f25cc6f3d', // steak
  'photo-1594007654729-407eedc4be65', // noodles
]

export const DEFAULT_FOOD_IMAGE =
  'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=800&auto=format&fit=crop'

// Get a consistent but varied image based on recipe ID
export function getUnsplashFoodImage(recipeId: number): string {
  const imageIndex = recipeId % foodImageIds.length
  const imageId = foodImageIds[imageIndex]
  return `https://images.unsplash.com/${imageId}?w=800&h=500&fit=crop&auto=format`
}

export function getRecipeImageUrl(
  images: string[] | null | undefined,
  recipeId: number
): string {
  return images?.[0] || getUnsplashFoodImage(recipeId)
}
