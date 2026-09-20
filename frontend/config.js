/**
 * Planwisely frontend deploy-time configuration (Step 10).
 *
 * PUBLIC values only — safe for the browser:
 *   SUPABASE_URL      public Supabase project URL
 *   SUPABASE_ANON_KEY public anon/publishable key (never the service-role key)
 *   API_BASE          Planwisely API base URL
 *
 * Replace the YOUR_* placeholders at deploy time (e.g. Vercel static deploy
 * replaces this file, or a small edge rewrite injects the values). The anon
 * key is designed for browser use; it is NOT a secret. No service-role key,
 * JWT secret, or any other backend credential may ever be placed here.
 */
window.PLANWISELY_CONFIG = {
  SUPABASE_URL: 'YOUR_SUPABASE_URL',
  SUPABASE_ANON_KEY: 'YOUR_SUPABASE_ANON_KEY',
  API_BASE: 'https://fin-advisor-sa6h.onrender.com',
};
