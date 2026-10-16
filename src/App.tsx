import { useState, useRef, useCallback, useEffect } from "react";
import ReactMarkdown from "react-markdown";

type Message = {
  role: "user" | "assistant";
  content: string;
};

type ChatState = {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
};

function getSessionId(): string {
  let sid = sessionStorage.getItem("gram_sahayak_session");
  if (!sid) {
    sid = crypto.randomUUID();
    sessionStorage.setItem("gram_sahayak_session", sid);
  }
  return sid;
}

export default function App() {
  const [state, setState] = useState<ChatState>({
    messages: [],
    isLoading: false,
    error: null,
  });
  const [input, setInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [speakingIndex, setSpeakingIndex] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [state.messages, scrollToBottom]);

  useEffect(() => {
    if (!("speechSynthesis" in window)) return;
    const synth = window.speechSynthesis;
    const onEnd = () => setSpeakingIndex(null);
    synth.addEventListener("end", onEnd);
    return () => {
      synth.removeEventListener("end", onEnd);
      synth.cancel();
    };
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || state.isLoading) return;

      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
        setSpeakingIndex(null);
      }

      const sessionId = getSessionId();
      const userMessage: Message = { role: "user", content: trimmed };
      const history = [...state.messages, userMessage];

      setState({
        messages: history,
        isLoading: true,
        error: null,
      });

      try {
        const resp = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: history.map((m) => ({ role: m.role, content: m.content })),
            session_id: sessionId,
          }),
        });

        if (!resp.ok || !resp.body) {
          throw new Error(`Server error: ${resp.status}`);
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let assistantText = "";

        // Add empty assistant message that we'll update
        setState((prev) => ({
          ...prev,
          messages: [
            ...prev.messages,
            { role: "assistant" as const, content: "" },
          ],
        }));

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          let currentEvent = "";
          for (const line of lines) {
            if (line.startsWith("event: ")) {
              currentEvent = line.slice(7).trim();
            } else if (line.startsWith("data: ") && currentEvent) {
              try {
                const data = JSON.parse(line.slice(6));
                if (currentEvent === "token" && data.token) {
                  assistantText += data.token;
                  setState((prev) => {
                    const msgs = [...prev.messages];
                    msgs[msgs.length - 1] = {
                      role: "assistant",
                      content: assistantText,
                    };
                    return { ...prev, messages: msgs };
                  });
                } else if (currentEvent === "replace" && data.text) {
                  assistantText = data.text;
                  setState((prev) => {
                    const msgs = [...prev.messages];
                    msgs[msgs.length - 1] = {
                      role: "assistant",
                      content: data.text,
                    };
                    return { ...prev, messages: msgs };
                  });
                } else if (currentEvent === "error") {
                  setState((prev) => ({
                    ...prev,
                    error: data.message || "Something went wrong",
                  }));
                }
              } catch {
                // skip
              }
              currentEvent = "";
            }
          }
        }
      } catch (err) {
        setState((prev) => ({
          ...prev,
          error: err instanceof Error ? err.message : "Connection failed",
        }));
      } finally {
        setState((prev) => ({ ...prev, isLoading: false }));
      }
    },
    [state.messages, state.isLoading]
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(input);
    setInput("");
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      audioChunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/wav",
        });
        stream.getTracks().forEach((t) => t.stop());

        try {
          const resp = await fetch("/api/transcribe", {
            method: "POST",
            headers: { "Content-Type": "application/octet-stream" },
            body: audioBlob,
          });
          if (!resp.ok) throw new Error("Transcription failed");
          const data = await resp.json();
          if (data.text) {
            sendMessage(data.text);
          }
        } catch {
          setState((prev) => ({
            ...prev,
            error: "Audio transcription failed. Please try typing instead.",
          }));
        }
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch {
      setState((prev) => ({
        ...prev,
        error: "Microphone access denied. Please allow microphone permissions.",
      }));
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  const readAloud = (text: string, index: number) => {
    if (!text.trim() || !("speechSynthesis" in window)) return;
    const synth = window.speechSynthesis;
    if (speakingIndex === index) {
      synth.cancel();
      setSpeakingIndex(null);
      return;
    }
    synth.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "hi-IN";
    utterance.rate = 0.9;
    setSpeakingIndex(index);
    synth.speak(utterance);
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>
          <span style={styles.emoji}>🌾</span> Gram Sahayak
        </h1>
        <p style={styles.subtitle}>Your AI assistant for rural India</p>
      </header>

      <div style={styles.chatContainer}>
        {state.messages.length === 0 && (
          <div style={styles.welcome}>
            <div style={styles.welcomeIcon}>🌾</div>
            <h2 style={styles.welcomeTitle}>Namaste! Main Gram Sahayak hu</h2>
            <p style={styles.welcomeText}>
              Ask me about government schemes, weather, or any question you have.
            </p>
            <div style={styles.suggestionGrid}>
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  style={styles.suggestionCard}
                  onClick={() => sendMessage(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {state.messages.map((msg, i) => (
          <div
            key={i}
            style={
              msg.role === "user" ? styles.userMessage : styles.assistantMessage
            }
          >
            <div
              style={{
                ...styles.messageBubble,
                background: msg.role === "user" ? "#0ea5e9" : "#f4f4f5",
                color: msg.role === "user" ? "#fff" : "#18181b",
              }}
            >
              <div style={styles.messageText}>
                {msg.role === "assistant" ? (
                  <ReactMarkdown
                    components={{
                      p: ({ children }) => <p style={styles.mdP}>{children}</p>,
                      strong: ({ children }) => (
                        <strong style={styles.mdStrong}>{children}</strong>
                      ),
                      ul: ({ children }) => <ul style={styles.mdUl}>{children}</ul>,
                      ol: ({ children }) => <ol style={styles.mdOl}>{children}</ol>,
                      li: ({ children }) => <li style={styles.mdLi}>{children}</li>,
                      h3: ({ children }) => <h3 style={styles.mdH3}>{children}</h3>,
                      h4: ({ children }) => <h4 style={styles.mdH4}>{children}</h4>,
                      blockquote: ({ children }) => (
                        <blockquote style={styles.mdBlockquote}>{children}</blockquote>
                      ),
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                ) : (
                  msg.content
                )}
              </div>
              {msg.role === "assistant" && msg.content && (
                <button
                  style={{
                    ...styles.readAloudBtn,
                    opacity: speakingIndex === i ? 1 : 0.6,
                  }}
                  onClick={() => readAloud(msg.content, i)}
                  title={speakingIndex === i ? "Stop reading" : "Read aloud"}
                >
                  {speakingIndex === i ? "⏹" : "🔊"}
                </button>
              )}
            </div>
          </div>
        ))}

        {state.isLoading && state.messages[state.messages.length - 1]?.role !== "assistant" && (
          <div style={styles.typingIndicator}>
            <span style={styles.typingDot} />
            <span style={styles.typingDot} />
            <span style={styles.typingDot} />
          </div>
        )}

        {state.error && (
          <div style={styles.errorBanner}>{state.error}</div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form style={styles.inputArea} onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="PM Kisan yojana kya hai?"
          style={styles.textInput}
          disabled={state.isLoading}
        />
        <button
          type="button"
          onClick={isRecording ? stopRecording : startRecording}
          style={{
            ...styles.micButton,
            background: isRecording ? "#ef4444" : styles.micButton.background,
          }}
          disabled={state.isLoading && !isRecording}
          title={isRecording ? "Stop recording" : "Speak your question"}
        >
          {isRecording ? "⏹" : "🎤"}
        </button>
        <button
          type="submit"
          style={{
            ...styles.sendButton,
            opacity: state.isLoading || !input.trim() ? 0.5 : 1,
          }}
          disabled={state.isLoading || !input.trim()}
        >
          Send
        </button>
      </form>
    </div>
  );
}

const SUGGESTIONS = [
  "PM Kisan yojana kya hai?",
  "Mumbai ka mausam kaisa hai?",
  "Ayushman Bharat ke baare mein batao",
  "MGNREGA mein kaam kaise milega?",
];

const styles: Record<string, React.CSSProperties> = {
  container: {
    maxWidth: "768px",
    margin: "0 auto",
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    background: "#fff",
    boxShadow: "0 0 40px rgba(0,0,0,0.05)",
  },
  header: {
    padding: "20px 24px",
    background: "linear-gradient(135deg, #0ea5e9, #0d9488)",
    color: "#fff",
    flexShrink: 0,
  },
  title: {
    fontSize: "24px",
    fontWeight: 700,
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  emoji: { fontSize: "28px" },
  subtitle: { fontSize: "14px", opacity: 0.9, marginTop: "4px" },
  chatContainer: {
    flex: 1,
    overflowY: "auto",
    padding: "24px",
    display: "flex",
    flexDirection: "column",
    gap: "16px",
  },
  welcome: {
    textAlign: "center",
    padding: "40px 20px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "12px",
  },
  welcomeIcon: { fontSize: "48px" },
  welcomeTitle: { fontSize: "20px", fontWeight: 600, color: "#0c4a6e" },
  welcomeText: { fontSize: "15px", color: "#71717a", maxWidth: "400px" },
  suggestionGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "12px",
    marginTop: "20px",
    width: "100%",
    maxWidth: "500px",
  },
  suggestionCard: {
    padding: "14px 16px",
    background: "#f0f9ff",
    border: "1px solid #bae6fd",
    borderRadius: "10px",
    fontSize: "14px",
    color: "#075985",
    cursor: "pointer",
    textAlign: "left",
    transition: "all 0.2s",
  },
  userMessage: { display: "flex", justifyContent: "flex-end", animation: "slideUp 0.3s ease-out" },
  assistantMessage: { display: "flex", justifyContent: "flex-start", animation: "slideUp 0.3s ease-out" },
  messageBubble: {
    maxWidth: "85%",
    padding: "12px 16px",
    borderRadius: "16px",
    display: "flex",
    alignItems: "flex-end",
    gap: "8px",
    flexWrap: "wrap",
  },
  messageText: { fontSize: "15px", lineHeight: 1.6, flex: 1, minWidth: 0 },
  mdP: { margin: 0, marginBottom: "8px" },
  mdP_last: { margin: 0 },
  mdStrong: { fontWeight: 600 },
  mdUl: { margin: "4px 0 8px 0", paddingLeft: "20px" },
  mdOl: { margin: "4px 0 8px 0", paddingLeft: "20px" },
  mdLi: { marginBottom: "4px", lineHeight: 1.5 },
  mdH3: { fontSize: "15px", fontWeight: 600, marginTop: "8px", marginBottom: "4px" },
  mdH4: { fontSize: "14px", fontWeight: 600, marginTop: "6px", marginBottom: "4px" },
  mdBlockquote: {
    borderLeft: "3px solid #d4d4d8",
    paddingLeft: "12px",
    margin: "8px 0",
    fontSize: "14px",
    opacity: 0.8,
  },
  readAloudBtn: {
    background: "none",
    border: "none",
    cursor: "pointer",
    fontSize: "18px",
    padding: "2px 4px",
    flexShrink: 0,
    transition: "opacity 0.2s",
  },
  typingIndicator: {
    display: "flex",
    gap: "4px",
    padding: "12px 16px",
    background: "#f4f4f5",
    borderRadius: "16px",
    width: "fit-content",
  },
  typingDot: {
    width: "8px",
    height: "8px",
    borderRadius: "50%",
    background: "#a1a1aa",
    animation: "bounce 1.4s infinite ease-in-out",
  },
  errorBanner: {
    background: "#fef2f2",
    border: "1px solid #fecaca",
    color: "#dc2626",
    padding: "10px 16px",
    borderRadius: "8px",
    fontSize: "14px",
  },
  inputArea: {
    display: "flex",
    gap: "8px",
    padding: "16px 20px",
    borderTop: "1px solid #e4e4e7",
    background: "#fff",
    flexShrink: 0,
  },
  textInput: {
    flex: 1,
    padding: "12px 16px",
    border: "1px solid #d4d4d8",
    borderRadius: "10px",
    fontSize: "15px",
    outline: "none",
    transition: "border-color 0.2s",
  },
  micButton: {
    padding: "12px 16px",
    background: "#0ea5e9",
    color: "#fff",
    border: "none",
    borderRadius: "10px",
    fontSize: "18px",
    cursor: "pointer",
    flexShrink: 0,
    transition: "background 0.2s",
  },
  sendButton: {
    padding: "12px 24px",
    background: "#0d9488",
    color: "#fff",
    border: "none",
    borderRadius: "10px",
    fontSize: "15px",
    fontWeight: 600,
    cursor: "pointer",
    flexShrink: 0,
    transition: "opacity 0.2s",
  },
};
