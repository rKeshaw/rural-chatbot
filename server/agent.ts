import {
  groqChat,
  tavilySearch,
  getWeather,
  isResponseSafe,
  GROQ_MODEL,
} from "./lib.ts";
import { searchKnowledgeBase } from "./db.ts";

type Message = { role: string; content: string };

const SYSTEM_PROMPT = `You are Gram Sahayak, a helpful AI assistant for rural India.

You help farmers and rural citizens with questions about:
- Government schemes (PM Kisan, Ayushman Bharat, MGNREGA, crop insurance, etc.)
- Weather and agriculture
- General knowledge and daily life questions

LANGUAGE RULES:
- Match the user's script. If they write in Devanagari, reply in Devanagari Hindi. If they write in Hinglish (Roman letters), reply in Hinglish.
- Never switch scripts mid-response.

RESPONSE RULES:
- Keep answers SHORT — maximum 4-5 lines for simple questions, max 8-10 lines for complex ones.
- Use simple bullet points only when listing steps or options. Do NOT add section headers, tables, or bold headings.
- Give the most important info first. Do not add a "Sources:" line or footnotes.
- Do not write long reports, essays, or information dumps. A farmer asking "kya hai?" wants a 2-3 line answer, not a full article.
- Be practical, friendly, and to the point.
- If the user asks about a place, give a 2-3 line summary, not a full district report.

ACCURACY RULES — CRITICAL:
- If you are given "Web search results" or "Knowledge base information" below, you MUST base your answer ONLY on that data. Do NOT add facts from your own memory.
- If the provided data does not contain the answer, say you don't have that information rather than guessing.
- Never make up numbers, dates, statistics, or details that are not in the provided data.`;

function detectScript(text: string): "devanagari" | "hinglish" {
  return /[\u0900-\u097F]/.test(text) ? "devanagari" : "hinglish";
}

async function routeQuery(
  messages: Message[]
): Promise<"general" | "web_search" | "weather" | "knowledge_base"> {
  const lastUser = [...messages].reverse().find((m) => m.role === "user");
  const userText = lastUser?.content ?? "";

  const routerPrompt = `You are a query router. Classify the user's message into exactly ONE category. Output only the category name, nothing else.

Categories:
- knowledge_base: Questions specifically about Indian government schemes by name (PM Kisan, Ayushman Bharat, PMJAY, MGNREGA, Kisan Credit Card, KCC, Fasal Bima Yojana, PMFBY, Soil Health Card)
- web_search: Questions about current events, real-time data, latest news, GDP, population, facts about specific places/cities/districts, geography, history, or anything requiring factual up-to-date information
- weather: Weather queries for a specific city or place
- general: Greetings, small talk, advice, or simple questions that don't need external data

User message: "${userText}"

Category:`;

  const resp = await groqChat(
    [{ role: "user", content: routerPrompt }],
    { max_tokens: 500, temperature: 0 }
  );
  if (!resp.ok) return "general";
  const data = (await resp.json()) as {
    choices: { message: { content: string } }[];
  };
  const result = data.choices[0]?.message?.content?.toLowerCase().trim() ?? "";

  if (result.includes("web")) return "web_search";
  if (result.includes("weather")) return "weather";
  if (result.includes("knowledge")) return "knowledge_base";
  return "general";
}

export async function handleChat(
  messages: Message[],
  sessionId: string,
  onToken: (token: string) => void
): Promise<string> {
  const route = await routeQuery(messages);
  const lastUser = [...messages].reverse().find((m) => m.role === "user");
  const userText = lastUser?.content ?? "";

  const userScript = detectScript(userText);
  const scriptInstruction =
    userScript === "devanagari"
      ? "\n\nIMPORTANT: The user wrote in Devanagari. You MUST reply in Devanagari Hindi script."
      : "\n\nIMPORTANT: The user wrote in Hinglish (Roman letters). You MUST reply in Hinglish (Roman letters), never Devanagari.";

  let systemPrompt = SYSTEM_PROMPT + scriptInstruction;
  let contextInfo = "";

  if (route === "web_search") {
    const searchResults = await tavilySearch(userText);
    if (searchResults) {
      contextInfo = `\n\nWeb search results (use ONLY this data, do not add your own facts):\n${searchResults}\n\nAnswer the user's question using ONLY the information above. If the search results don't contain the answer, say you don't have that information.`;
    }
  } else if (route === "weather") {
    const cityResp = await groqChat(
      [
        {
          role: "system",
          content:
            "Extract the city name from the user's weather query. Respond with ONLY the city name in English, nothing else.",
        },
        { role: "user", content: userText },
      ],
      { max_tokens: 500, temperature: 0 }
    );
    let city = userText;
    if (cityResp.ok) {
      const cityData = (await cityResp.json()) as {
        choices: { message: { content: string } }[];
      };
      const extracted = cityData.choices[0]?.message?.content?.trim();
      if (extracted && extracted.length > 0) city = extracted;
    }
    const weatherInfo = await getWeather(city);
    onToken(weatherInfo);
    return weatherInfo;
  } else if (route === "knowledge_base") {
    // If the query is in Devanagari, translate to English for KB matching
    let searchQuery = userText;
    if (userScript === "devanagari") {
      const transResp = await groqChat(
        [
          {
            role: "system",
            content:
              "Translate the user message to English. Output ONLY the English translation, nothing else.",
          },
          { role: "user", content: userText },
        ],
        { max_tokens: 500, temperature: 0 }
      );
      if (transResp.ok) {
        const transData = (await transResp.json()) as {
          choices: { message: { content: string } }[];
        };
        const translated = transData.choices[0]?.message?.content?.trim();
        if (translated && translated.length > 0) searchQuery = translated;
      }
    }
    const kbResults = await searchKnowledgeBase(searchQuery);
    if (kbResults) {
      contextInfo = `\n\nKnowledge base information (use ONLY this data, do not add your own facts):\n${kbResults}\n\nAnswer the user's question using ONLY the information above. If the knowledge base doesn't contain the answer, say you don't have that information.`;
    } else {
      const fallback =
        userScript === "devanagari"
          ? "माफ कीजिए, इस विषय पर मेरे पास अभी जानकारी नहीं है। आप अपने नजदीकी CSC (Common Service Center) से मदद ले सकते हैं।"
          : "Maaf kijiye, is vishay par mere paas abhi jankari nahi hai. Aap apne nearest CSC (Common Service Center) se madad le sakte hain.";
      onToken(fallback);
      return fallback;
    }
  }

  const fullMessages = [
    { role: "system", content: systemPrompt + contextInfo },
    ...messages,
  ];

  const resp = await groqChat(fullMessages, { stream: true, max_tokens: 600 });
  if (!resp.ok || !resp.body) {
    onToken("Maaf kijiye, abhi ek takneeki samasya aa gayi hai.");
    return "Maaf kijiye, abhi ek takneeki samasya aa gayi hai.";
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let fullResponse = "";
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith("data: ")) continue;
      const jsonStr = trimmed.slice(6);
      if (jsonStr === "[DONE]") continue;
      try {
        const parsed = JSON.parse(jsonStr) as {
          choices: { delta: { content?: string } }[];
        };
        const token = parsed.choices[0]?.delta?.content;
        if (token) {
          fullResponse += token;
          onToken(token);
        }
      } catch {
        // skip malformed chunks
      }
    }
  }

  return fullResponse;
}
