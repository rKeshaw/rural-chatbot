import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  // Load all env vars (including non-VITE_ prefixed) so server modules can read them
  const env = loadEnv(mode, process.cwd(), "");
  for (const [key, value] of Object.entries(env)) {
    if (!process.env[key]) process.env[key] = value;
  }

  return {
    server: {
      fs: { strict: false },
    },
    optimizeDeps: {
      exclude: ["@supabase/supabase-js"],
    },
    plugins: [
      {
        name: "gram-sahayak-api",
        async configureServer(server) {
          // Dynamic imports so env vars are set before server modules load
          const { handleChat } = await import("./server/agent.ts");
          const { getOrCreateProfile } = await import("./server/db.ts");
          const { groqTranscribe, isResponseSafe } = await import("./server/lib.ts");

          server.middlewares.use(async (req: any, res: any, next: any) => {
            if (!req.url?.startsWith("/api/")) return next();

            // CORS
            res.setHeader("Access-Control-Allow-Origin", "*");
            res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
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

            next();
          });
        },
      },
    ],
  };
});

function readBody(req: any): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString()));
    req.on("error", reject);
  });
}
