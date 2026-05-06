/**
 * MessageBubble.jsx
 * -----------------
 * Renders one chat turn. Style depends on the message type:
 *   user      — right-aligned, accent background
 *   assistant — left-aligned, includes the routing pill and sources panel
 *   loading   — left-aligned, animated dots, optionally shows the routing pill
 *               once the /classify call returns (stage="generating")
 *   error     — left-aligned, red border + message
 */

import SourcesPanel from "./SourcesPanel";

function RoutingPill({ routing }) {
  if (!routing) return null;
  const pct = Math.round((routing.confidence ?? 0) * 100);
  return (
    <div className="routing-pill" title="Predicted by the manual classifier">
      Routed to <strong>{shortenFilename(routing.predicted_source)}</strong>
      <span className="routing-pct">{pct}%</span>
    </div>
  );
}

function shortenFilename(name) {
  if (!name) return "?";
  // Long Samsung/HP filenames clutter the UI; use a friendly short form
  // when we recognize one, otherwise truncate.
  const friendly = {
    "db05a9.pdf": "LG washer",
    "c06184015.pdf": "HP printer",
    "cpd60205.pdf": "Epson printer",
    "Service-Manual-18.pdf": "Panasonic AC",
    "DA68-04752Q_FDR_RF6500C_3Door_EN_MES_CFR_260209.pdf": "Samsung fridge",
  };
  return friendly[name] || (name.length > 28 ? name.slice(0, 25) + "…" : name);
}

export default function MessageBubble({ message }) {
  if (message.type === "user") {
    return (
      <div className="bubble bubble-user">
        <div className="bubble-content">{message.content}</div>
      </div>
    );
  }

  if (message.type === "loading") {
    const label =
      message.stage === "routing"
        ? "Routing your question to the right manual…"
        : "Generating answer (2–3 min on CPU)…";
    return (
      <div className="bubble bubble-assistant bubble-loading">
        <RoutingPill routing={message.routing} />
        <div className="loading-row">
          <span className="dots">
            <span /><span /><span />
          </span>
          <span className="loading-label">{label}</span>
        </div>
      </div>
    );
  }

  if (message.type === "error") {
    return (
      <div className="bubble bubble-assistant bubble-error">
        <strong>Error.</strong> {message.content}
      </div>
    );
  }

  // type === "assistant"
  return (
    <div className="bubble bubble-assistant">
      <RoutingPill routing={message.routing} />
      <div className="bubble-content answer-content">{message.content}</div>
      {Array.isArray(message.sources) && message.sources.length > 0 && (
        <SourcesPanel sources={message.sources} />
      )}
    </div>
  );
}
