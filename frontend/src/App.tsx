import { FormEvent, KeyboardEvent, ReactNode, useEffect, useRef, useState } from "react";

type ResolutionSource = "LIVE_INFERENCE" | "LLM_CACHE";

interface ReplyMetadata {
  resolution_source: ResolutionSource;
  model: string;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  key_id: string | null;
  fallback_attempts: number;
}

interface ChatEntry {
  id: string;
  role: "user" | "assistant" | "error";
  content: string;
  metadata?: ReplyMetadata;
}

interface HealthResponse {
  status: string;
  keys_configured: number;
  cache_enabled: boolean;
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

const STARTER_PROMPTS = [
  "What is the capital of France?",
  "Explain how API key failover improves reliability in 3 bullet points.",
  "Write a short Python function to check if a number is prime.",
];

function createSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `s-${crypto.randomUUID().slice(0, 8)}`;
  }
  return `s-${Math.random().toString(36).slice(2, 10)}`;
}

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

function renderInlineFormatting(text: string, keyPrefix: string): ReactNode[] {
  const tokens = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return tokens.map((token, idx) => {
    const key = `${keyPrefix}-${idx}`;
    if (token.startsWith("**") && token.endsWith("**") && token.length > 4) {
      return <strong key={key}>{token.slice(2, -2)}</strong>;
    }
    if (token.startsWith("`") && token.endsWith("`") && token.length > 2) {
      return (
        <code key={key} className="inline-code">
          {token.slice(1, -1)}
        </code>
      );
    }
    return token;
  });
}

function FormattedReply({ content }: { content: string }) {
  const segments = content.split(/(```[\s\S]*?```)/g);

  return (
    <div className="message-markdown">
      {segments.map((segment, index) => {
        if (segment.startsWith("```") && segment.endsWith("```")) {
          const inner = segment.slice(3, -3);
          const firstNewline = inner.indexOf("\n");
          const language =
            firstNewline !== -1 ? inner.slice(0, firstNewline).trim() || "code" : "code";
          const codeBody =
            firstNewline !== -1 ? inner.slice(firstNewline + 1).trimEnd() : inner.trim();

          return (
            <div key={`code-${index}`} className="code-block">
              <div className="code-block-header">{language}</div>
              <pre>
                <code>{codeBody}</code>
              </pre>
            </div>
          );
        }

        const blocks = segment.split(/\n{2,}/).filter((b) => b.trim().length > 0);
        return blocks.map((block, bIdx) => {
          const lines = block.split("\n");
          const isList = lines.every((line) => /^\s*[-*]\s+/.test(line));
          if (isList) {
            return (
              <ul key={`list-${index}-${bIdx}`}>
                {lines.map((line, lIdx) => (
                  <li key={`li-${index}-${bIdx}-${lIdx}`}>
                    {renderInlineFormatting(
                      line.replace(/^\s*[-*]\s+/, ""),
                      `li-${index}-${bIdx}-${lIdx}`
                    )}
                  </li>
                ))}
              </ul>
            );
          }

          return (
            <p key={`p-${index}-${bIdx}`}>
              {lines.map((line, lIdx) => (
                <span key={`line-${index}-${bIdx}-${lIdx}`}>
                  {renderInlineFormatting(line, `p-${index}-${bIdx}-${lIdx}`)}
                  {lIdx < lines.length - 1 && <br />}
                </span>
              ))}
            </p>
          );
        });
      })}
    </div>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState<string>(() => createSessionId());
  const [messages, setMessages] = useState<ChatEntry[]>([]);
  const [input, setInput] = useState<string>("");
  const [systemPrompt, setSystemPrompt] = useState<string>("");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isSending, setIsSending] = useState<boolean>(false);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    let active = true;
    fetch(apiUrl("/health"))
      .then((res) => (res.ok ? res.json() : null))
      .then((data: HealthResponse | null) => {
        if (active) setHealth(data);
      })
      .catch(() => {
        if (active) setHealth(null);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  const startNewSession = () => {
    if (isSending) return;
    setSessionId(createSessionId());
    setMessages([]);
    inputRef.current?.focus();
  };

  const sendMessage = async (overrideMessage?: string) => {
    const rawText = overrideMessage ?? input;
    const trimmed = rawText.trim();
    if (isSending) return;

    // Clear input immediately on send to satisfy chat-input contract
    if (overrideMessage === undefined) {
      setInput("");
    }

    if (!trimmed) {
      return;
    }

    const userEntry: ChatEntry = {
      id: `user-${Date.now()}`,
      role: "user",
      content: trimmed,
    };

    setMessages((prev) => [...prev, userEntry]);
    setIsSending(true);

    try {
      const payload: Record<string, string> = {
        session_id: sessionId,
        message: trimmed,
      };
      if (systemPrompt.trim()) {
        payload.system_prompt = systemPrompt.trim();
      }

      const response = await fetch(apiUrl("/chat"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const body = await response.json().catch(() => null);

      if (!response.ok) {
        const detail =
          body && typeof body.detail === "string"
            ? body.detail
            : Array.isArray(body?.detail)
              ? body.detail.map((item: { msg?: string }) => item.msg || "Invalid input").join("; ")
              : `Request failed with status ${response.status}`;

        setMessages((prev) => [
          ...prev,
          {
            id: `error-${Date.now()}`,
            role: "error",
            content: `Error (${response.status}): ${detail}`,
          },
        ]);
        return;
      }

      const assistantEntry: ChatEntry = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: body.reply,
        metadata: body.metadata,
      };
      setMessages((prev) => [...prev, assistantEntry]);
    } catch (err) {
      const errorText =
        err instanceof Error ? err.message : "Unable to reach the backend server.";
      setMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          role: "error",
          content: `Network error: ${errorText}`,
        },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    void sendMessage();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  };

  return (
    <div className="app-layout">
      <header className="top-bar">
        <div className="brand-group">
          <span className="brand-badge">L</span>
          <div>
            <h1>Lyftr AI Chatbot</h1>
            <p className="session-label">
              Session: <code>{sessionId}</code>
            </p>
          </div>
        </div>

        <div className="header-controls">
          <div className="health-pill">
            <span
              className={`health-dot ${
                health?.status === "ok" && health.keys_configured > 0 ? "ok" : "warn"
              }`}
            />
            <span>
              {health
                ? `${health.keys_configured} key${health.keys_configured === 1 ? "" : "s"} · Cache ${
                    health.cache_enabled ? "ON" : "OFF"
                  }`
                : "Backend offline"}
            </span>
          </div>
          <button
            type="button"
            className="new-session-btn"
            onClick={startNewSession}
            disabled={isSending}
          >
            New Session
          </button>
        </div>
      </header>

      <section className="system-prompt-bar" aria-label="Session system prompt">
        <label htmlFor="system-prompt-input">System Prompt</label>
        <input
          id="system-prompt-input"
          data-testid="system-prompt-input"
          type="text"
          placeholder="Optional: override the default system prompt for this session..."
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
        />
      </section>

      <main className="chat-viewport">
        {messages.length === 0 ? (
          <div className="empty-state">
            <h2>Ask anything to test caching, failover &amp; token tracking</h2>
            <p>
              Each page load starts a fresh session. Repeat a first message across sessions to
              observe <code>LLM_CACHE</code> hits.
            </p>
            <div className="starter-grid">
              {STARTER_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  className="starter-card"
                  onClick={() => void sendMessage(prompt)}
                  disabled={isSending}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages-thread">
            {messages.map((entry) => {
              if (entry.role === "error") {
                return (
                  <div key={entry.id} className="chat-error-message" role="alert">
                    {entry.content}
                  </div>
                );
              }

              if (entry.role === "user") {
                return (
                  <article key={entry.id} className="message-row user">
                    <div className="avatar">You</div>
                    <div className="bubble user-bubble">
                      <p>{entry.content}</p>
                    </div>
                  </article>
                );
              }

              const meta = entry.metadata;
              return (
                <article
                  key={entry.id}
                  data-testid="assistant-message"
                  className="message-row assistant"
                >
                  <div className="avatar ai-avatar">AI</div>
                  <div className="bubble assistant-bubble">
                    <FormattedReply content={entry.content} />
                    {meta && (
                      <div className="metadata-bar">
                        <span
                          data-testid="meta-source"
                          className={`meta-badge ${
                            meta.resolution_source === "LLM_CACHE" ? "cache" : "live"
                          }`}
                        >
                          {meta.resolution_source}
                        </span>
                        <span className="meta-item">
                          Model: <strong data-testid="meta-model">{meta.model}</strong>
                        </span>
                        <span className="meta-item">
                          Latency:{" "}
                          <strong data-testid="meta-latency">{meta.latency_ms} ms</strong>
                        </span>
                        <span className="meta-item">
                          Input tokens:{" "}
                          <strong data-testid="meta-input-tokens">{meta.input_tokens}</strong>
                        </span>
                        <span className="meta-item">
                          Output tokens:{" "}
                          <strong data-testid="meta-output-tokens">{meta.output_tokens}</strong>
                        </span>
                        <span className="meta-item meta-extra">
                          Key: <strong>{meta.key_id ?? "cache"}</strong> · Fallbacks:{" "}
                          <strong>{meta.fallback_attempts}</strong>
                        </span>
                      </div>
                    )}
                  </div>
                </article>
              );
            })}

            {isSending && (
              <div className="message-row assistant">
                <div className="avatar ai-avatar">AI</div>
                <div className="bubble assistant-bubble typing">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      <form className="composer-bar" onSubmit={handleSubmit}>
        <div className="composer-box">
          <textarea
            ref={inputRef}
            data-testid="chat-input"
            rows={1}
            placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isSending}
          />
          <button
            type="submit"
            data-testid="send-button"
            className="send-btn"
            disabled={isSending || !input.trim()}
          >
            {isSending ? "Sending..." : "Send"}
          </button>
        </div>
      </form>
    </div>
  );
}
