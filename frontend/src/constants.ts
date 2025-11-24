const getApiUrl = () => {
  // En production (build), utiliser l'URL de production
  if (import.meta.env.PROD) {
    return 'https://recsysfood.onrender.com'
  }
  // En développement, utiliser l'URL locale
  return 'http://0.0.0.0:8000'
}

const env = {
  SUPABASE_URL: import.meta.env.VITE_SUPABASE_URL as string,
  SUPABASE_ANON_KEY: import.meta.env.VITE_SUPABASE_ANON_KEY as string,
  API_URL: getApiUrl(),
} as const

for (const [key, value] of Object.entries(env)) {
  if (!value) {
    throw new Error(`${key} is not set`)
  }
}

export default env