const env = {
  SUPABASE_URL: import.meta.env.VITE_SUPABASE_URL as string,
  SUPABASE_ANON_KEY: import.meta.env.VITE_SUPABASE_ANON_KEY as string,
  API_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
} as const

for (const [key, value] of Object.entries(env)) {
  if (!value) {
    throw new Error(`${key} is not set`)
  }
}

export default env