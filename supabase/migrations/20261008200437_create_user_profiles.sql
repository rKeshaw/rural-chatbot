/*
# Create user_profiles table for Gram Sahayak

1. New Tables
- `user_profiles`
  - `id` (uuid, primary key)
  - `session_id` (text, unique, identifies a user session)
  - `location` (text, nullable, user's city/region)
  - `language` (text, default 'Hinglish', preferred language)
  - `interests` (text[], default '{}', list of user interests)
  - `created_at` (timestamptz, default now())
  - `updated_at` (timestamptz, default now())

2. Security
- Enable RLS on `user_profiles`.
- This is a no-auth app (no sign-in screen), so allow anon + authenticated CRUD.
- All data is intentionally shared/public across sessions (single-tenant model).
*/

CREATE TABLE IF NOT EXISTS user_profiles (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id text UNIQUE NOT NULL,
    location text,
    language text NOT NULL DEFAULT 'Hinglish',
    interests text[] NOT NULL DEFAULT '{}',
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "anon_select_user_profiles" ON user_profiles;
CREATE POLICY "anon_select_user_profiles"
ON user_profiles FOR SELECT
TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "anon_insert_user_profiles" ON user_profiles;
CREATE POLICY "anon_insert_user_profiles"
ON user_profiles FOR INSERT
TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "anon_update_user_profiles" ON user_profiles;
CREATE POLICY "anon_update_user_profiles"
ON user_profiles FOR UPDATE
TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "anon_delete_user_profiles" ON user_profiles;
CREATE POLICY "anon_delete_user_profiles"
ON user_profiles FOR DELETE
TO anon, authenticated USING (true);
