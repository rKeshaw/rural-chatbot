import http from "node:http";
import { readFileSync, existsSync } from "node:fs";
import { join, extname } from "node:path";
import { handleChat } from "./agent.ts";
import { getOrCreateProfile } from "./db.ts";
import { groqTranscribe, isResponseSafe } from "./lib.ts";

const PORT = 3001;

const MIME: Record<string, string> = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

const STATIC_DIR = join(import.meta.dirname, "..", "dist");

const server = http.createServer(async (req, res) => {
  // CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "Content-Type, Authorization"
  );
  if (req.method === "OPTIONS") {
    res.writeHead(200);
    res.end();
    return;
  }

  // SSE chat endpoint
  if (req.url === "/api/chat" && req.method === "POST") {
    const body = await readBody(req);
    let parsed: { messages: { role: string; content: string }[]; session_id: string };
    try {
      parsed = JSON.parse(body);
    } catch {
      res.writeHead(400);
      res.end("Invalid JSON");
      return;
    }

    const { messages, session_id } = parsed;
    if (!session_id || !messages?.length) {
      res.writeHead(400);
      res.end("Missing session_id or messages");
      return;
    }

    // Ensure profile exists
    try {
      await getOrCreateProfile(session_id);
    } catch {
      // non-fatal
    }

    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });

    const send = (event: string, data: unknown) => {
      res.write(`event: ${event}\n`);
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    };

    try {
      const full = await handleChat(messages, session_id, (token) => {
        send("token", { token });
      });

      // Safety check
      const lastUser = [...messages].reverse().find((m) => m.role === "user");
      const safe = await isResponseSafe(lastUser?.content ?? "", full);
      if (!safe) {
        send("replace", {
          text: "Maaf kijiye, main is vishay par charcha nahi kar sakta.",
        });
      }

      send("done", {});
    } catch (err) {
      send("error", { message: String(err) });
    }

    res.end();
    return;
  }

  // Audio transcription endpoint
  if (req.url === "/api/transcribe" && req.method === "POST") {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk as Buffer);
    const audioBlob = new Blob([Buffer.concat(chunks)], { type: "audio/wav" });

    try {
      const text = await groqTranscribe(audioBlob);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ text }));
    } catch (err) {
      res.writeHead(500);
      res.end(JSON.stringify({ error: String(err) }));
    }
    return;
  }

  // Static file serving (for production)
  if (req.url && !req.url.startsWith("/api")) {
    let filePath = join(STATIC_DIR, req.url === "/" ? "index.html" : req.url);
    if (!existsSync(filePath)) {
      filePath = join(STATIC_DIR, "index.html");
    }
    if (existsSync(filePath)) {
      const content = readFileSync(filePath);
      res.writeHead(200, { "Content-Type": MIME[extname(filePath)] ?? "application/octet-stream" });
      res.end(content);
      return;
    }
  }

  res.writeHead(404);
  res.end("Not found");
});

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c as Buffer));
    req.on("end", () => resolve(Buffer.concat(chunks).toString()));
    req.on("error", reject);
  });
}

server.listen(PORT, () => {
  console.log(`Gram Sahayak server running on http://localhost:${PORT}`);
});
