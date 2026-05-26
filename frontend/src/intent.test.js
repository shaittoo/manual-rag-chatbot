/**
 * Unit tests for src/intent.js — the pre-network message routing heuristics.
 *
 * These behaviours directly control whether the frontend (a) answers locally,
 * (b) sends conversation history, or (c) treats the message as a fresh topic.
 * Getting them wrong causes the most visible UX bugs (history contamination,
 * greetings hitting the model), so they are worth pinning down.
 */

import { describe, it, expect } from "vitest";
import {
  normalizeMessage,
  isThankYouMessage,
  isGreetingMessage,
  hasExplicitNewTopic,
  isLikelyFollowUp,
} from "./intent";

describe("normalizeMessage", () => {
  it("lowercases, strips punctuation, and collapses whitespace", () => {
    expect(normalizeMessage("  Hello,   THERE!! ")).toBe("hello there");
  });

  it("keeps hyphens (model identifiers like et-4850)", () => {
    expect(normalizeMessage("ET-4850")).toBe("et-4850");
  });
});

describe("isGreetingMessage", () => {
  it("matches common greetings (incl. Filipino)", () => {
    for (const g of ["hi", "Hello", "hey there", "Good Morning", "kamusta"]) {
      expect(isGreetingMessage(g)).toBe(true);
    }
  });

  it("does not match real questions that merely start with a greeting word", () => {
    expect(isGreetingMessage("hi, my washer won't drain")).toBe(false);
  });
});

describe("isThankYouMessage", () => {
  it("matches thanks variants (incl. Filipino 'salamat')", () => {
    for (const t of ["thanks", "thank you", "ty", "salamat", "thanks so much"]) {
      expect(isThankYouMessage(t)).toBe(true);
    }
  });

  it("does not match a sentence that contains 'thanks'", () => {
    expect(isThankYouMessage("thanks but it still won't cool")).toBe(false);
  });
});

describe("hasExplicitNewTopic", () => {
  it("detects a named device/topic", () => {
    expect(hasExplicitNewTopic("how do I replace the ink cartridge?")).toBe(true);
    expect(hasExplicitNewTopic("my Epson printer jams")).toBe(true);
  });

  it("returns false for a vague pronoun question", () => {
    expect(hasExplicitNewTopic("what about that?")).toBe(false);
  });
});

describe("isLikelyFollowUp", () => {
  it("treats strong follow-up phrases as follow-ups", () => {
    expect(isLikelyFollowUp("Can I add laundry during that?")).toBe(true);
    expect(isLikelyFollowUp("what if it still happens?")).toBe(true);
  });

  it("treats short pronoun questions as follow-ups", () => {
    expect(isLikelyFollowUp("why?")).toBe(false); // no pronoun token
    expect(isLikelyFollowUp("what does that mean?")).toBe(true);
    expect(isLikelyFollowUp("can I do it?")).toBe(true);
  });

  it("does NOT treat a clearly new topic as a follow-up (avoids history contamination)", () => {
    // Even though it is short, naming a new device wins over the pronoun rule.
    expect(isLikelyFollowUp("how do I replace the ink cartridge?")).toBe(false);
    expect(isLikelyFollowUp("my Epson printer has a paper jam")).toBe(false);
  });

  it("does not over-trigger on a fresh, fully specified question", () => {
    expect(isLikelyFollowUp("My LG washer is not filling with water")).toBe(false);
  });
});
