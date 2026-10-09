/*
# Create knowledge_base_entries table for Gram Sahayak

1. New Tables
- `knowledge_base_entries`
  - `id` (uuid, primary key)
  - `source` (text, name of the source file or URL)
  - `content` (text, the knowledge chunk text)
  - `category` (text, nullable, e.g. 'government_schemes', 'agriculture', 'weather')
  - `created_at` (timestamptz, default now())

2. Security
- Enable RLS on `knowledge_base_entries`.
- No-auth app: allow anon + authenticated CRUD (single-tenant, public knowledge).
*/

CREATE TABLE IF NOT EXISTS knowledge_base_entries (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    content text NOT NULL,
    category text,
    created_at timestamptz DEFAULT now()
);

ALTER TABLE knowledge_base_entries ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "anon_select_kb_entries" ON knowledge_base_entries;
CREATE POLICY "anon_select_kb_entries"
ON knowledge_base_entries FOR SELECT
TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "anon_insert_kb_entries" ON knowledge_base_entries;
CREATE POLICY "anon_insert_kb_entries"
ON knowledge_base_entries FOR INSERT
TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "anon_update_kb_entries" ON knowledge_base_entries;
CREATE POLICY "anon_update_kb_entries"
ON knowledge_base_entries FOR UPDATE
TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "anon_delete_kb_entries" ON knowledge_base_entries;
CREATE POLICY "anon_delete_kb_entries"
ON knowledge_base_entries FOR DELETE
TO anon, authenticated USING (true);
