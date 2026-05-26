/**
 * App.jsx
 * -------
 * Orchestrator for the chat UI.
 *
 * Supports:
 * - Model dropdown: Transformers / Phi-3-mini or Ollama / Qwen 2.5 3B
 * - Conversational follow-up questions by sending recent chat history
 * - Custom quick replies for thank-you messages
 * - Custom quick replies for greetings like hello/hi
 * - Smarter history handling to avoid topic contamination
 */

import { useState, useRef } from "react";
import ChatWindow from "./components/ChatWindow";
import {
  isGreetingMessage,
  isLikelyFollowUp,
  isThankYouMessage,
} from "./intent";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const THANK_YOU_RESPONSES = [
  "You're welcome! Glad I could help.",
  "No problem! Ask me another manual question anytime.",
  "You're welcome. Let me know if you want to check another issue.",
  "Happy to help!",
  "Anytime! I’m here if you need help with another manual issue.",
];

const GREETING_RESPONSES = [
  "Hi! Ask me anything about the product manuals.",
  "Hello! What manual issue would you like help with?",
  "Hi there! You can ask me about troubleshooting, setup, maintenance, or product instructions.",
  "Hello! Tell me what appliance or device problem you want to check.",
  "Hi! I can help answer questions from the indexed manuals.",
];

function getRandomResponse(responses) {
  const index = Math.floor(Math.random() * responses.length);
  return responses[index];
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [isBusy, setIsBusy] = useState(false);
  const [generatorBackend, setGeneratorBackend] = useState("transformers");

  const nextId = useRef(1);

  function newId() {
    const id = nextId.current;
    nextId.current += 1;
    return id;
  }

  function buildHistory() {
    return messages
      .filter((m) => m.type === "user" || m.type === "assistant")
      .slice(-10)
      .map((m) => ({
        role: m.type === "user" ? "user" : "assistant",
        content: m.content,
      }));
  }

  function addLocalAssistantReply(userText, assistantText) {
    setMessages((prev) => [
      ...prev,
      {
        id: newId(),
        type: "user",
        content: userText,
      },
      {
        id: newId(),
        type: "assistant",
        content: assistantText,
        sources: [],
        routing: null,
        generatorBackend,
      },
    ]);
  }

  async function handleSubmit(query) {
    const trimmedQuery = query.trim();

    if (!trimmedQuery || isBusy) return;

    // Quick local response for greetings.
    // This avoids unnecessary /classify and /ask calls.
    if (isGreetingMessage(trimmedQuery)) {
      addLocalAssistantReply(
        trimmedQuery,
        getRandomResponse(GREETING_RESPONSES)
      );
      return;
    }

    // Quick local response for thank-you messages.
    // This avoids unnecessary /classify and /ask calls.
    if (isThankYouMessage(trimmedQuery)) {
      addLocalAssistantReply(
        trimmedQuery,
        getRandomResponse(THANK_YOU_RESPONSES)
      );
      return;
    }

    setIsBusy(true);

    /*
      Only send history if the current message is likely a follow-up.

      This prevents this problem:
      Q1: My printer keeps jamming.
      Q2: How do I replace the ink cartridge?
      → Q2 should not be forced to use paper-jam history.
    */
    const history = isLikelyFollowUp(trimmedQuery) ? buildHistory() : [];

    const loadingId = newId();

    setMessages((prev) => [
      ...prev,
      {
        id: newId(),
        type: "user",
        content: trimmedQuery,
      },
      {
        id: loadingId,
        type: "loading",
        stage: "routing",
        generatorBackend,
      },
    ]);

    let routing = null;

    // 1. Classify / route the query.
    try {
      const cRes = await fetch(`${API_URL}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: trimmedQuery,
          history,
        }),
      });

      if (!cRes.ok) {
        const errText = await cRes.text();
        throw new Error(
          `Classify failed (${cRes.status}): ${errText.slice(0, 200)}`
        );
      }

      routing = await cRes.json();
    } catch (e) {
      replaceLoadingWithError(
        loadingId,
        `Couldn't reach the classifier. ${e.message || e}`
      );
      setIsBusy(false);
      return;
    }

    setMessages((prev) =>
      prev.map((m) =>
        m.id === loadingId
          ? {
              ...m,
              stage: "generating",
              routing,
              generatorBackend,
            }
          : m
      )
    );

    // 2. Ask the RAG backend.
    try {
      const aRes = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: trimmedQuery,
          source: routing.predicted_source,
          top_k: 4,
          generator_backend: generatorBackend,
          history,
        }),
      });

      if (!aRes.ok) {
        const errText = await aRes.text();
        throw new Error(
          `Ask failed (${aRes.status}): ${errText.slice(0, 200)}`
        );
      }

      const result = await aRes.json();

      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingId
            ? {
                id: m.id,
                type: "assistant",
                content: result.answer,
                sources: result.sources,
                routing,
                generatorBackend,
              }
            : m
        )
      );
    } catch (e) {
      replaceLoadingWithError(
        loadingId,
        `Couldn't get an answer. ${e.message || e}`
      );
    } finally {
      setIsBusy(false);
    }
  }

  function replaceLoadingWithError(loadingId, msg) {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === loadingId
          ? {
              id: m.id,
              type: "error",
              content: msg,
            }
          : m
      )
    );
  }

  const modelLabel =
    generatorBackend === "ollama"
      ? "Ollama / Qwen 2.5 3B"
      : "Transformers / Phi-3-mini";

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <h1>Manu</h1>

          <div className="model-picker">
            <label htmlFor="generator-select">Model</label>
            <select
              id="generator-select"
              value={generatorBackend}
              onChange={(e) => setGeneratorBackend(e.target.value)}
              disabled={isBusy}
            >
              <option value="transformers">Transformers / Phi-3-mini</option>
              <option value="ollama">Ollama / Qwen 2.5 3B</option>
            </select>
          </div>
        </div>
      </header>

      <main className="app-main">
        <ChatWindow
          messages={messages}
          isBusy={isBusy}
          onSubmit={handleSubmit}
        />
      </main>

      <footer className="app-footer">
        <div className="app-footer-inner">
          <span>
            Backend: <code>{API_URL}</code>
          </span>
          <span className="hint">Current generator: {modelLabel}</span>
        </div>
      </footer>
    </div>
  );
}