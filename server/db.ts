import { createClient } from "@supabase/supabase-js";
import { SUPABASE_URL, SUPABASE_ANON_KEY } from "./lib.ts";

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

let entriesCache: Record<string, string>[] = [];

async function loadEntries(): Promise<Record<string, string>[]> {
  const { data, error } = await supabase
    .from("knowledge_base_entries")
    .select("*");
  if (error) throw error;
  return data ?? [];
}

export async function searchKnowledgeBase(
  query: string,
  k = 3
): Promise<string> {
  const entries = entriesCache.length ? entriesCache : await loadEntries();
  entriesCache = entries;
  if (!entries.length) return "";

  const queryWords = new Set(query.toLowerCase().split(/\s+/));
  const scored: { score: number; entry: Record<string, string> }[] = [];

  for (const entry of entries) {
    const contentWords = new Set(
      (entry.content || "").toLowerCase().split(/\s+/)
    );
    let score = 0;
    for (const w of queryWords) if (contentWords.has(w)) score++;
    if (score > 0) scored.push({ score, entry });
  }

  scored.sort((a, b) => b.score - a.score);
  const top = scored.slice(0, k);
  if (!top.length) return "";
  return top.map((s) => s.entry.content).join("\n\n");
}

export async function getOrCreateProfile(sessionId: string) {
  const { data: existing } = await supabase
    .from("user_profiles")
    .select("*")
    .eq("session_id", sessionId)
    .maybeSingle();

  if (existing) return existing;

  const profileData = {
    session_id: sessionId,
    location: null,
    language: "Hinglish",
    interests: [],
  };
  const { data: inserted } = await supabase
    .from("user_profiles")
    .insert(profileData)
    .select()
    .single();
  return inserted ?? profileData;
}
