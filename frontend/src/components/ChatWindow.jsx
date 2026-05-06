/**
 * ChatWindow.jsx
 * --------------
 * The scrolling list of messages plus the input box at the bottom.
 *
 * Auto-scrolls to the bottom whenever a new message is added or updated.
 * The input is a single-line textbox; Enter submits, Shift+Enter would
 * add a newline (we don't enable that here — single-question turns are
 * cleaner for a manual-Q&A demo).
 */

import { useEffect, useRef, useState } from "react";
import MessageBubble from "./MessageBubble";

export default function ChatWindow({ messages, isBusy, onSubmit }) {
  const [draft, setDraft] = useState("");
  const listRef = useRef(null);

  // Scroll to the latest message whenever the list changes.
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  function submit(e) {
    e.preventDefault();
    if (!draft.trim() || isBusy) return;
    onSubmit(draft);
    setDraft("");
  }

  return (
    <div className="chat-window">
      <div className="message-list" ref={listRef}>
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Ask a question about any of the indexed manuals.</p>
            <p className="examples">
              Try: <em>"how do I drain antifreeze from my washer?"</em>
              {" · "}
              <em>"my printer keeps jamming, what should I check?"</em>
            </p>
          </div>
        )}
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
      </div>

      <form className="composer" onSubmit={submit}>
        <input
          type="text"
          className="composer-input"
          placeholder={
            isBusy
              ? "Generating answer — please wait..."
              : "Ask a question…"
          }
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          disabled={isBusy}
          autoFocus
        />
        <button
          type="submit"
          className="composer-submit"
          disabled={isBusy || !draft.trim()}
        >
          {isBusy ? "Working…" : "Send"}
        </button>
      </form>
    </div>
  );
}
