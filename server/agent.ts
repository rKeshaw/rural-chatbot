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
You speak in simple Hindi and Hinglish (Hindi written in English letters).
You help farmers and rural citizens with questions about:
- Government schemes (PM Kisan, Ayushman Bharat, MGNREGA, crop insurance, etc.)
- Weather and agriculture
- General knowledge and daily life questions
Be concise, practical, and friendly. Always reply in Hindi or Hinglish.`;

async function routeQuery(
  messages: Message[]
): Promise<"general" | "web_search" | "weather" | "knowledge_base"> {
  const lastUser = [...messages].reverse().find((m) => m.role === "user");
  const userText = lastUser?.content ?? "";

  const routerPrompt = `You are a query router. Given the user's message, classify it into exactly one of these categories:
- "general" — greetings, small talk, general knowledge, advice
- "web_search" — current events, latest news, real-time information, things requiring up-to-date data
- "weather" — weather queries for a specific city or place
- "knowledge_base" — questions about Indian government schemes (PM Kisan, Ayushman Bharat, MGNREGA, Kisan Credit Card, crop insurance, soil health card, etc.)

Respond with ONLY the category name, nothing else.

User message: "${userText}"`;

  const resp = await groqChat(
    [{ role: "user", content: routerPrompt }],
    { max_tokens: 20, temperature: 0 }
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

  let systemPrompt = SYSTEM_PROMPT;
  let contextInfo = "";

  if (route === "web_search") {
    const searchResults = await tavilySearch(userText);
    contextInfo = `\n\nWeb search results:\n${searchResults}\n\nUse this information to answer the user's question.`;
  } else if (route === "weather") {
    const cityResp = await groqChat(
      [
        {
          role: "system",
          content:
            "Extract the city name from the user's weather query. Respond with ONLY the city name, nothing else.",
        },
        { role: "user", content: userText },
      ],
      { max_tokens: 20, temperature: 0 }
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
    const kbResults = await searchKnowledgeBase(userText);
    if (kbResults) {
      contextInfo = `\n\nKnowledge base information:\n${kbResults}\n\nUse this information to answer the user's question about the government scheme.`;
    } else {
      onToken(
        "Maaf kijiye, is vishay par mere paas abhi jankari nahi hai. Aap apne nearest CSC (Common Service Center) se madad le sakte hain."
      );
      return "Maaf kijiye, is vishay par mere paas abhi jankari nahi hai. Aap apne nearest CSC (Common Service Center) se madad le sakte hain.";
    }
  }

  const fullMessages = [
    { role: "system", content: systemPrompt + contextInfo },
    ...messages,
  ];

  const resp = await groqChat(fullMessages, { stream: true });
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
