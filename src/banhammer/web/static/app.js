"use strict";

const FIELDS = ["name", "mana_cost", "type_line", "oracle_text", "power", "toughness", "loyalty"];
const SEARCH_DELAY_MS = 200;
const RISK_LABELS = { ok: "Allowed", watch: "Under watch", likely_ban: "Likely ban" };

// Invented cards to try the forge quickly, plus existing ones loaded by name.
const CUSTOM_EXAMPLES = [
  {
    label: "Echo Weaver",
    card: {
      name: "Echo Weaver", mana_cost: "{2}{U}{R}", type_line: "Enchantment — Aura",
      oracle_text: "Enchant creature\nEnchanted creature has \"{T}: Create a token that's a copy of this creature, " +
        "except it has haste. Exile it at the beginning of the next end step.\"",
    },
  },
  {
    label: "Grove Warden",
    card: {
      name: "Grove Warden", mana_cost: "{2}{G}", type_line: "Creature — Elf Druid",
      oracle_text: "Reach\nWhen Grove Warden enters, you gain 3 life.", power: "2", toughness: "3",
    },
  },
  {
    label: "Ashen Bargain",
    card: {
      name: "Ashen Bargain", mana_cost: "{B}", type_line: "Sorcery",
      oracle_text: "As an additional cost to cast this spell, sacrifice a creature.\n" +
        "Draw cards equal to its power. Each opponent loses that much life.",
    },
  },
];
const EXISTING_EXAMPLES = ["Splinter Twin", "Hogaak, Arisen Necropolis", "Storm Crow"];

const form = document.getElementById("card-form");
const preview = document.getElementById("card-preview");
const verdict = document.getElementById("verdict");
const judgeButton = document.getElementById("judge-button");
const searchInput = document.getElementById("search-input");
const searchResults = document.getElementById("search-results");

// A card loaded from the catalog is sent whole (it may have faces) until the user edits the form.
let loadedCard = null;

/* ---------- DOM helpers (text only: report content never goes through innerHTML) ---------- */

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child !== null && child !== undefined) node.append(child);
  }
  return node;
}

/** Replace a node's children, skipping the null placeholders of optional parts. */
function setChildren(node, ...children) {
  node.replaceChildren(...children.filter((child) => child !== null && child !== undefined));
}

function pip(symbol) {
  const plain = symbol.replace("/", "");
  const color = /^[WUBRGC]$/.test(plain) ? plain : (symbol.match(/[WUBRG]/) || [""])[0];
  return el("span", { class: `pip pip-${color}`, title: `{${symbol}}` }, symbol === "T" ? "↷" : symbol);
}

/** Text with every {X} mana or tap symbol drawn as a pip. */
function withSymbols(text) {
  const parts = [];
  let last = 0;
  for (const match of text.matchAll(/\{([^}]+)\}/g)) {
    parts.push(text.slice(last, match.index), pip(match[1]));
    last = match.index + match[0].length;
  }
  parts.push(text.slice(last));
  return parts;
}

function frameColor(manaCost) {
  const colors = new Set(manaCost.match(/[WUBRG]/g) || []);
  if (colors.size === 0) return "var(--frame-none)";
  if (colors.size > 1) return "var(--frame-multi)";
  return `var(--frame-${[...colors][0]})`;
}

/* ---------- the forge ---------- */

function formCard() {
  const data = new FormData(form);
  const card = {};
  for (const field of FIELDS) {
    const value = String(data.get(field) || "").trim();
    if (value) card[field] = value;
  }
  return card;
}

function frontFace(card) {
  return card.card_faces && card.card_faces.length ? { ...card.card_faces[0], name: card.name } : card;
}

function fillForm(card) {
  const face = frontFace(card);
  for (const field of FIELDS) form.elements[field].value = face[field] || card[field] || "";
  renderPreview();
}

function renderPreview() {
  // An existing card shows its real image until the user edits it into a new card.
  const image = loadedCard && loadedCard.image;
  preview.classList.toggle("has-image", Boolean(image));
  if (image) {
    setChildren(preview, el("img", { src: image, alt: loadedCard.name }));
    return;
  }
  const card = formCard();
  const mana = card.mana_cost || "";
  preview.style.setProperty("--frame", frameColor(mana));
  const stats = card.loyalty || (card.power || card.toughness ? `${card.power || "?"}/${card.toughness || "?"}` : "");
  setChildren(preview,
    el("div", { class: "card-title" }, el("span", {}, card.name || "Unnamed card"), el("span", { class: "mana" }, withSymbols(mana))),
    el("div", { class: "card-art", "aria-hidden": "true" }),
    el("div", { class: "card-type" }, el("span", {}, card.type_line || "Type line")),
    el("div", { class: "card-text" }, withSymbols(card.oracle_text || "")),
    stats ? el("div", { class: "card-pt" }, stats) : null,
  );
}

form.addEventListener("input", (event) => {
  if (event.target.name === "format") return;
  loadedCard = null;
  renderPreview();
});

function useCard(card) {
  loadedCard = card;
  fillForm(card);
}

/* ---------- search ---------- */

let searchTimer;
let activeIndex = -1;
let lastResults = [];

async function findCards(query) {
  const response = await fetch(`/api/cards?q=${encodeURIComponent(query)}`);
  if (!response.ok) throw new Error(`search failed (${response.status})`);
  return (await response.json()).cards;
}

function closeResults() {
  searchResults.hidden = true;
  searchInput.setAttribute("aria-expanded", "false");
  activeIndex = -1;
}

function showResults(cards) {
  lastResults = cards;
  activeIndex = -1;
  searchResults.replaceChildren(...cards.map((card, index) => {
    const option = el("li", { role: "option", id: `result-${index}`, "aria-selected": "false" },
      card.image ? el("img", { class: "thumb", src: card.image, alt: "" }) : el("span", { class: "thumb" }),
      el("span", { class: "result-name" }, card.name),
      el("span", { class: "type" }, frontFace(card).type_line || card.type_line || ""));
    option.addEventListener("mousedown", (event) => {
      event.preventDefault();
      pickResult(index);
    });
    return option;
  }));
  searchResults.hidden = cards.length === 0;
  searchInput.setAttribute("aria-expanded", String(cards.length > 0));
}

function pickResult(index) {
  useCard(lastResults[index]);
  searchInput.value = lastResults[index].name;
  closeResults();
}

function highlight(index) {
  activeIndex = index;
  [...searchResults.children].forEach((option, i) => option.setAttribute("aria-selected", String(i === index)));
  searchInput.setAttribute("aria-activedescendant", index >= 0 ? `result-${index}` : "");
}

searchInput.addEventListener("input", () => {
  clearTimeout(searchTimer);
  const query = searchInput.value;
  if (!query.trim()) return closeResults();
  searchTimer = setTimeout(() => {
    findCards(query).then(showResults).catch((error) => showError(error.message));
  }, SEARCH_DELAY_MS);
});

searchInput.addEventListener("keydown", (event) => {
  if (searchResults.hidden) return;
  if (event.key === "ArrowDown") highlight(Math.min(activeIndex + 1, lastResults.length - 1));
  else if (event.key === "ArrowUp") highlight(Math.max(activeIndex - 1, 0));
  else if (event.key === "Enter" && activeIndex >= 0) pickResult(activeIndex);
  else if (event.key === "Escape") closeResults();
  else return;
  event.preventDefault();
});
searchInput.addEventListener("blur", closeResults);

/* ---------- quick trials ---------- */

function exampleChip(tag, label, onPick) {
  const chip = el("button", { type: "button", class: "chip" }, el("span", { class: "tag" }, tag), label);
  chip.addEventListener("click", onPick);
  return chip;
}

document.getElementById("examples").append(
  ...CUSTOM_EXAMPLES.map(({ label, card }) => exampleChip("forged", label, () => useCard({ ...card }))),
  ...EXISTING_EXAMPLES.map((name) => exampleChip("existing", name, async () => {
    try {
      const match = (await findCards(name)).find((card) => card.name === name);
      if (!match) throw new Error(`${name} is not in the card catalog`);
      useCard(match);
    } catch (error) {
      showError(error.message);
    }
  })),
);

/* ---------- the verdict ---------- */

function showError(message) {
  verdict.replaceChildren(el("p", { class: "error", role: "alert" }, message));
}

function sourceEntry(id, sources) {
  const source = sources[id];
  if (!source) return el("p", { class: "footnote" }, `Uncited source ${id}`);
  const isRule = source.namespace === "rules";
  // Card hits read "Name | Type line | Mana cost | Rules text".
  const parts = isRule ? [] : source.text.split(" | ");
  const body = parts.length >= 4
    ? [el("span", { class: "src-type" }, parts[1], " ", el("span", { class: "mana" }, withSymbols(parts[2]))),
       withSymbols(parts.slice(3).join(" | "))]
    : withSymbols(source.text);
  return el("details", { class: "source" },
    el("summary", {}, el("span", { class: "kind" }, isRule ? "RULE" : "CARD"), isRule ? id : source.name),
    el("div", { class: source.image ? "source-body with-image" : "source-body" },
      source.image ? el("img", { src: source.image, alt: source.name, loading: "lazy" }) : null,
      el("div", {}, body)));
}

function sourceList(ids, sources) {
  return ids.length ? el("div", { class: "sources" }, ids.map((id) => sourceEntry(id, sources))) : null;
}

function section(title, ...content) {
  return el("div", { class: "section" }, el("h3", {}, title), content);
}

function renderVerdict({ card, image, format, report, sources, dropped_claims: dropped }) {
  const risk = report.risk;
  const confidence = Math.round(report.confidence * 100);
  const combos = report.combos.length
    ? report.combos.map((combo) => el("div", { class: "item" },
        el("p", {}, combo.description),
        combo.sequence.length ? el("ol", {}, combo.sequence.map((step) => el("li", {}, step))) : null,
        sourceList(combo.card_ids, sources)))
    : el("p", { class: "none" }, "No combos found.");
  const issues = report.wording_issues.length
    ? report.wording_issues.map((issue) => el("div", { class: "item" },
        el("p", {}, issue.description), sourceList(issue.rule_ids, sources)))
    : el("p", { class: "none" }, "The wording holds up.");
  const power = report.power_assessment;

  setChildren(verdict,
    el("div", { class: "seal-row" },
      // The seal is stamped over the judged card when it has a real image.
      el("div", { class: image ? "trial with-image" : "trial" },
        image ? el("img", { src: image, alt: card.name }) : null,
        el("div", { class: `seal ${risk}` }, RISK_LABELS[risk] || risk)),
      el("div", { class: "seal-meta" },
        el("h3", {}, card.name),
        el("span", { class: "format-tag" }, `Judged for ${format}`),
        el("div", { class: "confidence" }, `Confidence ${confidence}%`,
          el("div", { class: "confidence-bar" }, el("span", { style: `width: ${confidence}%` }))))),
    report.risk_rationale ? el("p", { class: "rationale" }, report.risk_rationale) : null,
    section("Combos", combos),
    section("Wording", issues),
    section("Power", el("p", {}, power.summary || "No assessment."),
      power.comparable_ids.length ? el("p", { class: "footnote" }, "Comparable cards:") : null,
      sourceList(power.comparable_ids, sources)),
    dropped ? el("p", { class: "footnote" }, `${dropped} claim(s) discarded for lacking a valid citation.`) : null,
  );
}

function showDeliberation(name) {
  const started = Date.now();
  const clock = el("span", {}, "0s");
  verdict.replaceChildren(el("div", { class: "deliberating" },
    el("div", { class: "spinner", "aria-hidden": "true" }),
    el("p", {}, `The council is weighing ${name}...`),
    clock));
  const timer = setInterval(() => { clock.textContent = `${Math.round((Date.now() - started) / 1000)}s`; }, 1000);
  return () => clearInterval(timer);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const card = loadedCard || formCard();
  const format = new FormData(form).get("format");
  judgeButton.disabled = true;
  const stopClock = showDeliberation(card.name);
  try {
    const response = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ card, format }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `evaluation failed (${response.status})`);
    renderVerdict(payload);
  } catch (error) {
    showError(error.message);
  } finally {
    stopClock();
    judgeButton.disabled = false;
  }
});

renderPreview();
