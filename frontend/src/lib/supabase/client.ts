import { createClient } from '@supabase/supabase-js'
import env from '@/constants'

const options = {
    db: {
        schema: 'public',
    },
    auth: {
        autoRefreshToken: true,
        persistSession: true,
        detectSessionInUrl: true
    }
}
export const supabase = createClient(env.SUPABASE_URL, env.SUPABASE_ANON_KEY, options)