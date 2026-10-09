// Text-only chat shell for the Home Service Agent.
//
// The user types in the composer, hits send, and the text is POSTed to
// /api/classify. The response is rendered as a chat bubble in the transcript
// and a lead-result card in the side panel.

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"),
  phone: $("phone"),
  callerInfo: $("callerInfo"),
  resetBtn: $("resetBtn"),
  transcript: $("transcript"),
  result: $("result"),
  resultBody: $("resultBody"),
  summary: $("summary"),
  hint: $("hint"),
  composer: $("composer"),
  textInput: $("textInput"),
  sendBtn: $("sendBtn"),
};

const state = {
  turns: [],
  callClosed: false,
  busy: false,           // backend round-trip in flight
};

// ── UI helpers ──────────────────────────────────────────────────────────────

function setStatus(text, kind = "idle") {
  els.status.textContent = text;
  els.status.dataset.state = kind;
}

function setBusy(busy) {
  state.busy = busy;
  els.textInput.disabled = busy;
  els.sendBtn.disabled = busy;
}

function renderTurns() {
  els.transcript.replaceChildren();
  for (const t of state.turns) els.transcript.appendChild(turnNode(t.speaker, t.text));
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function turnNode(speaker, text) {
  const div = document.createElement("div");
  div.className = `turn ${speaker}`;
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = speaker;
  const body = document.createElement("span");
  body.textContent = text;
  div.appendChild(who);
  div.appendChild(body);
  return div;
}

function pushTurn(speaker, text) {
  const trimmed = (text || "").trim();
  if (!trimmed) return;
  state.turns.push({ speaker, text: trimmed });
  renderTurns();
}

function appendDtDd(parent, dtText, ddContent) {
  const dt = document.createElement("dt");
  dt.textContent = dtText;
  const dd = document.createElement("dd");
  if (ddContent instanceof Node) dd.appendChild(ddContent);
  else dd.textContent = ddContent == null ? "—" : String(ddContent);
  parent.appendChild(dt);
  parent.appendChild(dd);
}

function urgencyPill(level) {
  const span = document.createElement("span");
  const lvl = (level || "medium").toLowerCase();
  span.className = `pill ${lvl}`;
  span.textContent = level || "—";
  return span;
}

function showResult(pred) {
  els.result.hidden = false;
  els.resultBody.replaceChildren();
  appendDtDd(els.resultBody, "Category", pred.category);
  appendDtDd(els.resultBody, "Subcategory", pred.subcategory);
  appendDtDd(els.resultBody, "Urgency", urgencyPill(pred.urgency_level || pred.urgency));
  appendDtDd(els.resultBody, "Confidence", pred.confidence != null ? `${(pred.confidence * 100).toFixed(0)}%` : "—");
  // Provider match — show first matched provider details
  const providers = pred.merged_providers || pred.provider_matches || [];
  if (providers.length > 0) {
    const p = providers[0];
    const pd = document.createElement("div");
    pd.style.cssText = "font-size:12px;line-height:1.5";
    pd.appendChild(document.createTextNode(p.name || "—"));
    if (p.phone) { pd.appendChild(document.createElement("br")); pd.appendChild(document.createTextNode(p.phone)); }
    if (p.address || p.formattedAddress) { pd.appendChild(document.createElement("br")); pd.appendChild(document.createTextNode(p.address || p.formattedAddress)); }
    if (p.rating != null) { pd.appendChild(document.createElement("br")); pd.appendChild(document.createTextNode(`\u2605 ${p.rating} `)); }
    if (p.open_now != null) { pd.appendChild(document.createTextNode(p.open_now ? "\u2022 Open" : "\u2022 Closed")); }
    appendDtDd(els.resultBody, "Provider", pd);
    if (providers.length > 1) {
      appendDtDd(els.resultBody, "Alt. providers", `${providers.length - 1} more`);
    }
  } else {
    appendDtDd(els.resultBody, "Provider match", "—");
  }
  appendDtDd(els.resultBody, "Lead quality", pred.lead_quality != null ? String(pred.lead_quality) : "—");
  appendDtDd(els.resultBody, "Needs human review", pred.needs_human_review ? "yes" : "no");
  appendDtDd(els.resultBody, "Needs clarification", pred.needs_clarification ? "yes" : "no");
  els.summary.textContent = pred.call_summary || "";
}

function showCallerProfile(profile) {
  if (!profile) {
    els.callerInfo.hidden = true;
    els.callerInfo.replaceChildren();
    return;
  }
  els.callerInfo.hidden = false;
  els.callerInfo.replaceChildren();
  const name = profile.caller_name || profile.name || "unnamed";
  els.callerInfo.appendChild(document.createTextNode("User: "));
  const nameEl = document.createElement("b");
  nameEl.textContent = name;
  els.callerInfo.appendChild(nameEl);
  if (profile.phone || profile.phone_number) {
    els.callerInfo.appendChild(document.createTextNode(" \u00b7 "));
    els.callerInfo.appendChild(document.createTextNode(profile.phone || profile.phone_number));
  }
  if (profile.address) {
    els.callerInfo.appendChild(document.createElement("br"));
    els.callerInfo.appendChild(document.createTextNode("Address: "));
    const addrEl = document.createElement("b");
    addrEl.textContent = profile.address;
    els.callerInfo.appendChild(addrEl);
  }
}

// ── Backend round-trip ──────────────────────────────────────────────────────

async function classifyAndRespond() {
  setBusy(true);
  setStatus("thinking", "thinking");

  const phone = els.phone.value.trim() || null;
  let pred;
  try {
    const resp = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ turns: state.turns, caller_phone: phone }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }
    pred = await resp.json();
  } catch (e) {
    console.error(e);
    setStatus("backend error", "error");
    setBusy(false);
    return;
  }

  showCallerProfile(pred.caller_profile);
  showResult(pred);

  if (pred.search_expanded) {
    pushTurn("agent", "I'm expanding the search to cover a wider area…");
  }

  const needsMore = pred.needs_clarification && pred.clarification_question;

  if (needsMore) {
    const question = pred.clarification_question;
    pushTurn("agent", question);
    setBusy(false);
  } else {
    state.callClosed = true;
    const providerName = pred.merged_providers && pred.merged_providers.length > 0
      ? pred.merged_providers[0].name || pred.merged_providers[0].displayName || "a provider"
      : "";
    const dispatched = providerName
      ? ` Matching with ${providerName}.`
      : pred.needs_human_review
      ? " Routing this for human review."
      : "";
    const spoken = (pred.call_summary || "Got it.") + dispatched;
    pushTurn("agent", spoken);
    setBusy(false);
    setStatus("call closed", "idle");
  }
}

// ── Wiring ──────────────────────────────────────────────────────────────────

els.resetBtn.addEventListener("click", () => resetAll());

// Text composer
els.composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = els.textInput.value.trim();
  if (!text || state.busy) return;

  if (state.callClosed) await resetAll();

  if (state.turns.length === 0) pushTurn("agent", "Home Service Agent, what's going on?");

  pushTurn("caller", text);
  els.textInput.value = "";
  classifyAndRespond();
});

async function resetAll() {
  state.turns = [];
  state.callClosed = false;
  state.busy = false;
  setBusy(false);
  renderTurns();
  els.result.hidden = true;
  els.summary.textContent = "";
  els.resultBody.replaceChildren();
  els.textInput.value = "";
  showCallerProfile(null);
  setStatus("idle", "idle");
}

// ── Cosmetic: iPhone-style statusbar clock + call timer ─────────────────
// Purely visual — does not touch any agent state or backend calls.

const sbClock = document.getElementById("sbClock");
const callTimer = document.getElementById("callTimer");
const phoneShell = document.querySelector(".phone");
let timerStartedAt = 0;

function tickClock() {
  if (!sbClock) return;
  const d = new Date();
  let h = d.getHours();
  const m = d.getMinutes().toString().padStart(2, "0");
  if (h === 0) h = 12;
  else if (h > 12) h -= 12;
  sbClock.textContent = `${h}:${m}`;
}
tickClock();
setInterval(tickClock, 15000);

function fmtElapsed(ms) {
  const total = Math.floor(ms / 1000);
  const mm = Math.floor(total / 60).toString().padStart(2, "0");
  const ss = (total % 60).toString().padStart(2, "0");
  return `${mm}:${ss}`;
}

function tickTimer() {
  const live = !state.callClosed && state.turns.length > 0;
  if (phoneShell) {
    phoneShell.dataset.state = live ? "in-call" : (state.callClosed ? "ended" : "idle");
  }
  if (!callTimer) return;
  if (live) {
    if (!timerStartedAt) timerStartedAt = Date.now();
    callTimer.hidden = false;
    callTimer.textContent = fmtElapsed(Date.now() - timerStartedAt);
  } else {
    timerStartedAt = 0;
    callTimer.hidden = true;
    callTimer.textContent = "00:00";
  }
}
setInterval(tickTimer, 1000);