/**
 * intent.js
 * ---------
 * Pure, UI-free message-intent helpers used by App.jsx.
 *
 * These decide, before any network call, whether an incoming message is:
 *   - a greeting        -> answer locally, skip /classify and /ask
 *   - a thank-you       -> answer locally, skip /classify and /ask
 *   - a follow-up       -> include recent chat history when calling the backend
 *
 * They are extracted here (away from React state and rendering) so they can be
 * unit-tested in isolation. This is the most heuristic-heavy logic in the
 * frontend, so it is also the most worth testing.
 */

export function normalizeMessage(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function isThankYouMessage(text) {
  const normalized = normalizeMessage(text);

  const thanksPatterns = [
    "thanks",
    "thank you",
    "thank u",
    "thankyou",
    "ty",
    "thx",
    "tnx",
    "salamat",
    "okay thanks",
    "ok thanks",
    "okay thank you",
    "ok thank you",
    "thank you so much",
    "thanks so much",
    "many thanks",
  ];

  return thanksPatterns.includes(normalized);
}

export function isGreetingMessage(text) {
  const normalized = normalizeMessage(text);

  const greetingPatterns = [
    "hi",
    "hello",
    "hey",
    "heyy",
    "hello there",
    "hi there",
    "hey there",
    "good morning",
    "good afternoon",
    "good evening",
    "morning",
    "afternoon",
    "evening",
    "kamusta",
    "kumusta",
  ];

  return greetingPatterns.includes(normalized);
}

export function hasExplicitNewTopic(text) {
  const normalized = normalizeMessage(text);

  const topicKeywords = [
    // Brand/model/device identifiers
    "epson",
    "et 4850",
    "et-4850",
    "ecotank",
    "hp",
    "laserjet",

    // Epson / ink printer topics
    "ink",
    "ink tank",
    "ink tanks",
    "refill ink",
    "refilling ink",
    "cartridge",
    "ink cartridge",
    "maintenance box",
    "printhead",
    "print head",
    "nozzle",

    // HP / laser printer topics
    "toner",
    "toner cartridge",
    "document feeder",
    "scanner",
    "scan",
    "copy",
    "fax",

    // Washer identifiers/topics
    "washer",
    "washing machine",
    "lg washer",
    "laundry",
    "rinse",
    "spin",
    "rinse spin",
    "antifreeze",
    "detergent",
    "drain",
    "drain hose",
    "leak",
    "leaking",
    "tub clean",

    // Refrigerator identifiers/topics
    "refrigerator",
    "fridge",
    "freezer",
    "samsung refrigerator",
    "ice maker",
    "water dispenser",
    "water filter",
    "cooling",
    "temperature",
    "door alarm",
  ];

  return topicKeywords.some((keyword) => normalized.includes(keyword));
}

export function isLikelyFollowUp(text) {
  const normalized = normalizeMessage(text);

  /*
    Strong follow-up phrases should use history.

    Examples:
    Previous: "How do I drain antifreeze from my washer?"
    Current:  "Can I add laundry during that?"
    → use history

    Previous: "My Epson printer has a paper jam."
    Current:  "What if it still happens?"
    → use history
  */
  const strongFollowUpPatterns = [
    "what if",
    "what about",
    "how about",
    "after that",
    "during that",
    "while doing that",
    "can i do that",
    "can i add",
    "should i do that",
    "is that okay",
    "is it okay",
    "what next",
    "next step",
    "still happens",
    "still happening",
    "still does not work",
    "still doesn't work",
    "does not work",
    "doesn't work",
    "again",
  ];

  if (strongFollowUpPatterns.some((pattern) => normalized.includes(pattern))) {
    return true;
  }

  /*
    If the user clearly names a new device/model/topic, treat it as a fresh question.

    Examples:
    Previous: "My printer keeps jamming."
    Current:  "How do I replace the ink cartridge?"
    → do NOT use paper-jam history

    Previous: "How do I drain antifreeze from my washer?"
    Current:  "My Epson printer has a paper jam."
    → do NOT use washer history
  */
  if (hasExplicitNewTopic(normalized)) {
    return false;
  }

  /*
    Short vague pronoun-based messages are likely follow-ups.

    Examples:
    - "why?"
    - "how?"
    - "what does that mean?"
    - "can I do it?"
  */
  const tokens = normalized.split(" ").filter(Boolean);
  const pronouns = ["it", "that", "this", "those", "them"];

  const hasPronoun = tokens.some((token) => pronouns.includes(token));
  const isShort = tokens.length <= 8;

  if (isShort && hasPronoun) {
    return true;
  }

  return false;
}
