import http from "node:http";

const GROQ_API_KEY = process.env.GROQ_API_KEY || "";
const TAVILY_API_KEY = process.env.TAVILY_API_KEY || "";
const OPENWEATHER_API_KEY =
  process.env.OPENWEATHERMAP_API_KEY || process.env.OPENWEATHER_API_KEY || "";
const SUPABASE_URL =
  process.env.VITE_SUPABASE_URL || process.env.SUPABASE_URL || "";
const SUPABASE_ANON_KEY =
  process.env.VITE_SUPABASE_ANON_KEY || process.env.SUPABASE_ANON_KEY || "";

const GROQ_MODEL = "openai/gpt-oss-20b";
const WHISPER_MODEL = "whisper-large-v3";

export {
  GROQ_API_KEY,
  TAVILY_API_KEY,
  OPENWEATHER_API_KEY,
  SUPABASE_URL,
  SUPABASE_ANON_KEY,
  GROQ_MODEL,
  WHISPER_MODEL,
};

export async function groqChat(
  messages: { role: string; content: string }[],
  options: { stream?: boolean; max_tokens?: number; temperature?: number } = {}
): Promise<Response> {
  return fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model: GROQ_MODEL,
      messages,
      stream: options.stream ?? false,
      max_tokens: options.max_tokens,
      temperature: options.temperature ?? 0.7,
    }),
  });
}

export async function groqTranscribe(audioBlob: Blob): Promise<string> {
  const formData = new FormData();
  formData.append("file", audioBlob, "audio.wav");
  formData.append("model", WHISPER_MODEL);
  formData.append("language", "hi");

  const resp = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
    method: "POST",
    headers: { Authorization: `Bearer ${GROQ_API_KEY}` },
    body: formData,
  });
  if (!resp.ok) throw new Error(`Transcription failed: ${resp.status}`);
  const data = (await resp.json()) as { text: string };
  return data.text.trim();
}

export async function tavilySearch(query: string): Promise<string> {
  const resp = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      api_key: TAVILY_API_KEY,
      query,
      search_depth: "basic",
      max_results: 3,
    }),
  });
  if (!resp.ok) throw new Error(`Tavily search failed: ${resp.status}`);
  const data = (await resp.json()) as {
    results: { url: string; content: string }[];
  };
  return data.results
    .map((r) => `Source: ${r.url}\nContent: ${r.content}`)
    .join("\n\n");
}

export async function getWeather(city: string): Promise<string> {
  const url = `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(
    city
  )}&appid=${OPENWEATHER_API_KEY}&units=metric`;
  const resp = await fetch(url);
  if (!resp.ok) {
    return `An error occurred while fetching weather for ${city}.`;
  }
  const data = (await resp.json()) as {
    weather: { description: string }[];
    main: { temp: number; feels_like: number };
  };
  return `Weather data for ${city}: Condition is ${data.weather[0].description}, Temperature is ${data.main.temp}°C, Feels like ${data.main.feels_like}°C.`;
}

export async function isResponseSafe(
  userQuery: string,
  assistantResponse: string
): Promise<boolean> {
  try {
    const resp = await groqChat(
      [
        {
          role: "system",
          content:
            "You are a safety classifier. Analyze the conversation and respond with only one word: 'safe' or 'unsafe'. Unsafe means the response contains harmful, dangerous, illegal, or inappropriate content.",
        },
        {
          role: "user",
          content: `User: ${userQuery}\nAssistant: ${assistantResponse}\n\nIs the assistant's response safe or unsafe? Reply with only 'safe' or 'unsafe'.`,
        },
      ],
      { max_tokens: 100 }
    );
    if (!resp.ok) return true;
    const data = (await resp.json()) as {
      choices: { message: { content: string } }[];
    };
    const result = data.choices[0]?.message?.content?.toLowerCase().trim() ?? "";
    return !result.includes("unsafe");
  } catch {
    return true;
  }
}

export { http };
