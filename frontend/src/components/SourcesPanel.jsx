/**
 * SourcesPanel.jsx
 * ----------------
 * Collapsible disclosure for retrieved sources. Closed by default to keep
 * the chat readable; expanding shows filename + page + score + snippet for
 * each retrieved chunk that fed the LLM's answer.
 *
 * Score interpretation:
 *   - In V1 the score is cosine similarity (range 0..1).
 *   - In V2/V3 the score is the cross-encoder logit (typically -10..+15).
 *   We render whatever the backend sends; the relative ordering is what
 *   matters for the user, not the absolute value.
 */

import { useState } from "react";

export default function SourcesPanel({ sources }) {
  const [open, setOpen] = useState(false);
  const count = sources.length;

  return (
    <details
      className={`sources-panel ${open ? "is-open" : ""}`}
      open={open}
      onToggle={(e) => setOpen(e.currentTarget.open)}
    >
      <summary>
        {open ? "Hide" : "Show"} {count} source{count === 1 ? "" : "s"}
      </summary>
      <ol className="sources-list">
        {sources.map((s, i) => (
          <li key={i} className="source-item">
            <div className="source-meta">
              <span className="source-name">{s.source}</span>
              <span className="source-page">page {s.page}</span>
              <span className="source-score">score {fmtScore(s.score)}</span>
            </div>
            <div className="source-snippet">{s.snippet}</div>
          </li>
        ))}
      </ol>
    </details>
  );
}

function fmtScore(score) {
  if (typeof score !== "number") return String(score);
  return Math.abs(score) >= 1
    ? score.toFixed(2)
    : score.toFixed(4);
}
