/**
 * App.jsx
 * -------
 * Orchestrator for the chat UI.
 *
 * Conversation flow:
 *   1. User submits a query.
 *   2. POST /classify → predicts which manual the query is about (~5s).
 *      We show "Routing to <filename>..." as soon as this returns so the
 *      user sees the system thinking, rather than staring at a blank
 *      spinner for 2-3 minutes.
 *   3. POST /ask {query, source: predicted} → runs RAG with the predicted
 *      manual as the source filter (~2-3 minutes on CPU).
 *   4. Final answer is added to the chat alongside its retrieved sources.
 *
 * We deliberately call /classify then /ask separately rather than the
 * single-shot /ask_auto endpoint so the user gets intermediate progress
 * (routing decision visible after ~5s, before the long generation).
 */

import { useState, useRef } from "react";
import ChatWindow from "./components/ChatWindow";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  // Each message: { id, type, ... }
  //   user:      { id, type: "user", content }
  //   assistant: { id, type: "assistant", content, sources, routing }
  //   loading:   { id, type: "loading", stage, routing? }
  //   error:     { id, type: "error", content }
  const [messages, setMessages] = useState([]);
  const [isBusy, setIsBusy] = useState(false);
  const nextId = useRef(1);

  function newId() {
    const id = nextId.current;
    nextId.current += 1;
    return id;
  }

  async function handleSubmit(query) {
    if (!query.trim() || isBusy) return;
    setIsBusy(true);

    // 1. Push user message + a loading placeholder.
    const loadingId = newId();
    setMessages((prev) => [
      ...prev,
      { id: newId(), type: "user", content: query },
      { id: loadingId, type: "loading", stage: "routing" },
    ]);

    // 2. Classify (fast).
    let routing = null;
    try {
      const cRes = await fetch(`${API_URL}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
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

    // Update the loading bubble with routing info + new stage.
    setMessages((prev) =>
      prev.map((m) =>
        m.id === loadingId
          ? { ...m, stage: "generating", routing }
          : m
      )
    );

    // 3. Ask (slow).
    try {
      const aRes = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          source: routing.predicted_source,
          top_k: 4,
        }),
      });
      if (!aRes.ok) {
        const errText = await aRes.text();
        throw new Error(
          `Ask failed (${aRes.status}): ${errText.slice(0, 200)}`
        );
      }
      const result = await aRes.json();

      // Replace the loading bubble with the real assistant message.
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingId
            ? {
                id: m.id,
                type: "assistant",
                content: result.answer,
                sources: result.sources,
                routing,
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
          ? { id: m.id, type: "error", content: msg }
          : m
      )
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <h1>Manu</h1>
          <span className="tagline">
            Manual Q&amp;A — auto-routed across 5 appliance manuals
          </span>
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
          <span className="hint">
            Each answer takes ~2–3 min on CPU (Phi-3-mini, greedy decoding).
          </span>
        </div>
      </footer>
    </div>
  );
}
