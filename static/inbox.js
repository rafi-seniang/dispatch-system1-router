"use strict";
// Pilah inbox: loads the shift's messages, triages them in one call, and keeps the desk's work (read, assigned) in
// this page. Assignments are not sent anywhere: the app has no case system behind it yet.

const TEAM_NAMES = { appointment: "Appointments", billing: "Billing", results: "Results", medication: "Medication", emergency: "IGD" };
const URGENCY_NAMES = ["Routine", "Today", "Immediate"];
const LANGUAGE_NAMES = { id: "Bahasa Indonesia", en: "English" };
const COMPLAINT_LINE = 0.5;

const state = {
  messages: [],          // the shift's messages in arrival order, test messages appended
  triage: new Map(),     // message_id -> { action, decision }
  handled: new Map(),    // message_id -> { team, at }
  read: new Set(),
  view: "action",        // action | done | all
  query: "",
  selectedId: null,
  settings: null,        // { engine, gate, escalateAt, latencyMs, count, triagedAt }
  loading: true,
  error: null,
  testCount: 0,
};

const byId = (id) => document.getElementById(id);

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (character) => `&#${character.charCodeAt(0)};`);
}

function clockNow() {
  return new Date().toTimeString().slice(0, 5);
}

function probability(value) {
  return value.toFixed(2);
}

function plural(count, singular, pluralForm) {
  return `${count} ${count === 1 ? singular : pluralForm}`;
}

function needsAction(message) {
  const triaged = state.triage.get(message.message_id);
  return Boolean(triaged) && triaged.action !== "auto_route" && !state.handled.has(message.message_id);
}

function currentTeam(message) {
  const handled = state.handled.get(message.message_id);
  return handled ? handled.team : state.triage.get(message.message_id).decision.department;
}

function matchesQuery(message) {
  if (!state.query) return true;
  const query = state.query.toLowerCase();
  return message.text.toLowerCase().includes(query) || message.sender.toLowerCase().includes(query);
}

function messagesInScope() {
  return state.messages.filter((message) => state.triage.has(message.message_id) && matchesQuery(message));
}

function hasAction(message, action) {
  return needsAction(message) && state.triage.get(message.message_id).action === action;
}

/** The list as [heading, messages] groups: what to act on in arrival order (the longest wait first), done newest first. */
function groupsInView() {
  const inScope = messagesInScope();
  const done = [...inScope].reverse().filter((message) => !needsAction(message));
  if (state.view === "done") return [[null, done]];
  const groups = [
    ["Escalate now", inScope.filter((message) => hasAction(message, "escalate_now"))],
    ["Needs a person", inScope.filter((message) => hasAction(message, "human_review"))],
  ];
  if (state.view === "all") groups.push(["Done", done]);
  return groups.filter(([, messages]) => messages.length);
}

/* ---------- title row and sidebar ---------- */

function renderFinding() {
  const finding = byId("finding");
  if (state.loading) return;
  if (state.error) {
    finding.textContent = "The messages could not be triaged.";
    return;
  }
  const triaged = state.messages.filter((message) => state.triage.has(message.message_id));
  const open = triaged.filter(needsAction);
  const escalated = open.filter((message) => state.triage.get(message.message_id).action === "escalate_now").length;
  const needPerson = open.length - escalated;
  const settled = triaged.length - open.length;
  // Test messages carry the real clock, so the span is the earliest and latest time, not the first and last row
  const times = triaged.map((message) => message.received_at);
  const first = times.reduce((earliest, time) => (time < earliest ? time : earliest));
  const last = times.reduce((latest, time) => (time > latest ? time : latest));
  const openPart = open.length
    ? `${escalated} escalated and ${needPerson} ${needPerson === 1 ? "needs" : "need"} a person`
    : "Nothing needs a person";
  finding.textContent = `${openPart}. ${settled} routed or handled, of ${triaged.length} messages from ${first} to ${last}.`;
  byId("nav-count").textContent = open.length || "";
}

function renderStamp() {
  const settings = state.settings;
  byId("stamp").textContent = settings
    ? `Triaged by ${settings.engine} at ${settings.triagedAt}, ${settings.count} messages in ${Math.round(settings.latencyMs)} ms`
    : "";
  byId("rerun").disabled = state.loading;
}

function renderEngineFacts() {
  const settings = state.settings;
  if (!settings) return;
  byId("engine-facts").innerHTML = `
    <div><dt>Engine</dt><dd>${escapeHtml(settings.engine)}</dd></div>
    <div><dt>Gate</dt><dd>${probability(settings.gate)}</dd></div>
    <div><dt>Escalate from</dt><dd>urgency ${settings.escalateAt.toFixed(2)}</dd></div>`;
}

/* ---------- the list ---------- */

function renderTabs() {
  const inScope = messagesInScope();
  const counts = { action: inScope.filter(needsAction).length, all: inScope.length };
  counts.done = counts.all - counts.action;
  const tabs = [["action", "Needs action"], ["done", "Done"], ["all", "All"]];
  byId("tabs").innerHTML = tabs
    .map(([view, label]) => `<button role="tab" type="button" data-view="${view}" aria-selected="${state.view === view}">${label}<span class="count">${state.loading ? "" : counts[view]}</span></button>`)
    .join("");
}

function gateMini(topProbability) {
  const gate = state.settings.gate;
  const below = topProbability < gate ? " below" : "";
  return `<span class="gate-mini${below}" title="Top team ${probability(topProbability)}, gate ${probability(gate)}">
    <i style="width:${(topProbability * 100).toFixed(1)}%"></i><b style="left:${(gate * 100).toFixed(1)}%"></b></span>`;
}

function statusPill(message) {
  if (hasAction(message, "escalate_now")) return '<span class="pill neg">Escalate now</span>';
  if (hasAction(message, "human_review")) return '<span class="pill warn">Needs a person</span>';
  return "";
}

function messageRow(message) {
  const decision = state.triage.get(message.message_id).decision;
  const topProbability = Math.max(...Object.values(decision.department_probabilities));
  const unread = needsAction(message) && !state.read.has(message.message_id) ? " unread" : "";
  return `<li><button type="button" class="message-row${unread}" data-id="${escapeHtml(message.message_id)}" aria-current="${message.message_id === state.selectedId}">
    <span class="sender">${escapeHtml(message.sender)}</span><span class="time">${escapeHtml(message.received_at)}</span>
    <span class="preview">${escapeHtml(message.text)}</span>
    <span class="meta">${gateMini(topProbability)}<span class="team">${TEAM_NAMES[currentTeam(message)]}</span></span>
  </button></li>`;
}

function renderList() {
  const list = byId("messages");
  if (state.loading) {
    const placeholder = `<li><div class="message-row"><span class="skeleton" style="width:140px"></span><span class="skeleton" style="width:36px"></span>
      <span class="skeleton" style="width:220px"></span><span class="skeleton" style="width:96px"></span></div></li>`;
    list.innerHTML = placeholder.repeat(7);
    return;
  }
  if (state.error) {
    list.innerHTML = `<li class="list-empty error"><p>Couldn't triage the inbox: ${escapeHtml(state.error)}.</p>
      <p><button class="card-link" type="button" id="retry">Retry</button></p></li>`;
    return;
  }
  const groups = groupsInView();
  if (!groups.length || !groups[0][1].length) {
    const sentence = state.query ? `No messages match "${escapeHtml(state.query)}".` : state.view === "action" ? "Every message is routed or handled." : "No messages here yet.";
    list.innerHTML = `<li class="list-empty empty">${sentence}</li>`;
    return;
  }
  list.innerHTML = groups
    .map(([heading, messages]) => (heading ? `<li class="group-head"><span>${heading}</span><span>${messages.length}</span></li>` : "") + messages.map(messageRow).join(""))
    .join("");
}

/* ---------- the open message ---------- */

function verdictSentence(message) {
  const { action, decision } = state.triage.get(message.message_id);
  const { gate, escalateAt } = state.settings;
  const top = decision.department_probabilities[decision.department];
  const urgencyLevel = URGENCY_NAMES[Math.round(decision.urgency_score)].toLowerCase();
  let reason;
  if (action === "escalate_now") {
    reason = decision.department === "emergency"
      ? "Escalated because the model chose IGD."
      : `Escalated because urgency ${decision.urgency_score.toFixed(2)} is at or above ${escalateAt.toFixed(2)}.`;
  } else if (action === "human_review") {
    reason = `Sent to a person because ${probability(top)} is under the ${probability(gate)} gate.`;
  } else {
    reason = `Routed because ${probability(top)} clears the ${probability(gate)} gate.`;
  }
  const complaint = decision.complaint_probability >= COMPLAINT_LINE ? ` It reads as a complaint (${probability(decision.complaint_probability)}).` : "";
  return `The model sends this to <strong>${TEAM_NAMES[decision.department]}</strong> (${probability(top)}) and rates it
    <strong>${urgencyLevel}</strong> (${decision.urgency_score.toFixed(2)} of 2). ${reason}${complaint}`;
}

function assignBlock(message) {
  const { action, decision } = state.triage.get(message.message_id);
  const handled = state.handled.get(message.message_id);
  if (handled) {
    return `<div class="assign-row"><span class="done-note">${handled.team === "emergency" ? "Sent to IGD" : `Assigned to ${TEAM_NAMES[handled.team]}`} at ${handled.at}</span>
      <button class="btn ghost" type="button" data-reopen>Reopen</button></div>`;
  }
  const choice = message.choice || decision.department;
  const buttons = Object.keys(TEAM_NAMES)
    .map((team) => `<button type="button" data-team="${team}" aria-pressed="${team === choice}">${TEAM_NAMES[team]}</button>`)
    .join("");
  const routed = action === "auto_route";
  const unchanged = routed && choice === decision.department;
  const primaryText = unchanged ? `Already with ${TEAM_NAMES[choice]}` : choice === "emergency" ? "Send to IGD" : `${routed ? "Move" : "Assign"} to ${TEAM_NAMES[choice]}`;
  return `<div class="assign">
    <h3>${routed ? `Routed to ${TEAM_NAMES[decision.department]} by the model. Change team` : "Send to"}</h3>
    <div class="segmented team-choice" role="group" aria-label="Team">${buttons}</div>
    <div class="assign-row"><button class="btn primary" type="button" data-assign ${unchanged ? "disabled" : ""}>${primaryText}</button></div>
  </div>`;
}

function renderOpen() {
  const target = byId("open-message");
  const message = state.messages.find((candidate) => candidate.message_id === state.selectedId);
  if (state.loading || !message || !state.triage.has(message.message_id)) {
    target.innerHTML = state.loading
      ? '<span class="skeleton" style="width:180px;height:24px"></span><span class="skeleton" style="margin-top:24px;width:100%;height:96px"></span>'
      : "";
    return;
  }
  const language = LANGUAGE_NAMES[message.language] || "Language not set";
  const status = statusPill(message);
  target.innerHTML = `
    <button class="btn ghost back" type="button" data-back>Back to the list</button>
    <div class="open-head">
      <div><h2>${escapeHtml(message.sender)}</h2><p>WhatsApp at ${escapeHtml(message.received_at)}, ${language}</p></div>
      ${status}
    </div>
    <p class="bubble">${escapeHtml(message.text)}</p>
    <p class="verdict">${verdictSentence(message)}</p>
    ${assignBlock(message)}`;
}

/* ---------- why: the three answers ---------- */

function bar(name, value, isChosen, tickAt) {
  const tick = tickAt === null ? "" : `<b style="left:${(tickAt * 100).toFixed(1)}%"></b>`;
  return `<li class="${isChosen ? "chosen" : ""}"><span class="name">${name}</span><span class="value">${probability(value)}</span>
    <span class="track"><i style="width:${(value * 100).toFixed(1)}%"></i>${tick}</span></li>`;
}

function renderWhy() {
  const target = byId("why");
  const message = state.messages.find((candidate) => candidate.message_id === state.selectedId);
  if (state.loading || !message || !state.triage.has(message.message_id)) {
    target.innerHTML = '<h3>Why the model decided this</h3><span class="skeleton" style="height:180px"></span>';
    return;
  }
  const { decision } = state.triage.get(message.message_id);
  const { gate, escalateAt } = state.settings;
  const chosenLevel = Math.round(decision.urgency_score);
  const teamBars = Object.keys(TEAM_NAMES)
    .map((team) => bar(TEAM_NAMES[team], decision.department_probabilities[team], team === decision.department, team === decision.department ? gate : null))
    .join("");
  const urgencyBars = decision.urgency_probabilities.map((value, level) => bar(URGENCY_NAMES[level], value, level === chosenLevel, null)).join("");
  const isComplaint = decision.complaint_probability >= COMPLAINT_LINE;
  target.innerHTML = `
    <h3>Why the model decided this</h3>
    <section class="answer"><header><strong>Team</strong><span>gate ${probability(gate)}, a person below it</span></header>
      <ul class="bars">${teamBars}</ul></section>
    <section class="answer"><header><strong>Urgency</strong><span>${decision.urgency_score.toFixed(2)} of 2, escalates from ${escalateAt.toFixed(2)}</span></header>
      <ul class="bars">${urgencyBars}</ul></section>
    <section class="answer"><header><strong>Complaint</strong><span>yes from ${probability(COMPLAINT_LINE)}</span></header>
      <ul class="bars">${bar(isComplaint ? "Yes, a complaint" : "Not a complaint", decision.complaint_probability, isComplaint, COMPLAINT_LINE)}</ul></section>
    <dl class="facts">
      <div><dt>Received</dt><dd>${escapeHtml(message.received_at)}</dd></div>
      <div><dt>Message</dt><dd class="mono">${escapeHtml(message.message_id)}</dd></div>
    </dl>`;
}

function render() {
  renderFinding();
  renderStamp();
  renderEngineFacts();
  renderTabs();
  renderList();
  renderOpen();
  renderWhy();
  if (!state.loading && !state.error) {
    const inScope = messagesInScope();
    console.assert(inScope.filter(needsAction).length + inScope.filter((message) => !needsAction(message)).length === inScope.length, "tab counts must add up");
  }
}

/* ---------- actions ---------- */

function select(messageId) {
  state.selectedId = messageId;
  state.read.add(messageId);
  document.body.classList.add("show-message");
  history.replaceState(null, "", `#message=${encodeURIComponent(messageId)}`);  // a link to this message
  render();
}

function linkedMessageId() {
  const linked = new URLSearchParams(location.hash.slice(1)).get("message");
  return state.messages.some((message) => message.message_id === linked) ? linked : null;
}

function firstMessageToOpen() {
  const groups = groupsInView();
  return groups.length && groups[0][1].length ? groups[0][1][0].message_id : null;
}

async function postJson(url, body) {
  const response = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!response.ok) throw new Error(`the triage service answered ${response.status}`);
  return response.json();
}

async function triageAll() {
  state.loading = true;
  state.error = null;
  render();
  try {
    const response = await postJson("/v1/triage/batch", { messages: state.messages.map((message) => message.text) });
    state.messages.forEach((message, index) => state.triage.set(message.message_id, response.results[index]));
    state.settings = {
      engine: response.engine,
      gate: response.review_probability_threshold,
      escalateAt: response.escalate_urgency_score,
      latencyMs: response.latency_ms,
      count: state.messages.length,
      triagedAt: clockNow(),
    };
    state.selectedId = state.selectedId || firstMessageToOpen();
  } catch (error) {
    state.error = error.message;
  }
  state.loading = false;
  render();
}

async function loadShift() {
  try {
    const response = await fetch("/static/inbox.json");
    if (!response.ok) throw new Error(`the inbox file answered ${response.status}`);
    const shift = await response.json();
    state.messages = shift.messages;
    byId("scope-line").textContent = shift.line;
    byId("scope-shift").textContent = shift.shift;
  } catch (error) {
    state.loading = false;
    state.error = error.message;
    render();
    return;
  }
  await triageAll();
  const linked = linkedMessageId();
  if (linked && !state.error) {
    // Open the tab that holds the linked message, so it is visible in the list as well
    state.view = needsAction(state.messages.find((message) => message.message_id === linked)) ? "action" : "done";
    select(linked);
  }
}

byId("tabs").addEventListener("click", (event) => {
  const tab = event.target.closest("[data-view]");
  if (!tab) return;
  state.view = tab.dataset.view;
  render();
});

byId("messages").addEventListener("click", (event) => {
  if (event.target.closest("#retry")) {
    triageAll();
    return;
  }
  const row = event.target.closest("[data-id]");
  if (row) select(row.dataset.id);
});

byId("open-message").addEventListener("click", (event) => {
  const message = state.messages.find((candidate) => candidate.message_id === state.selectedId);
  if (event.target.closest("[data-back]")) {
    document.body.classList.remove("show-message");
    history.replaceState(null, "", location.pathname);
    return;
  }
  const teamButton = event.target.closest("[data-team]");
  if (teamButton) {
    message.choice = teamButton.dataset.team;
    renderOpen();
    return;
  }
  if (event.target.closest("[data-assign]")) {
    const decision = state.triage.get(message.message_id).decision;
    state.handled.set(message.message_id, { team: message.choice || decision.department, at: clockNow() });
    delete message.choice;
    const next = firstMessageToOpen();
    if (state.view === "action" && next) {
      select(next);
    } else {
      render();
    }
    return;
  }
  if (event.target.closest("[data-reopen]")) {
    state.handled.delete(message.message_id);
    render();
  }
});

byId("composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  const textBox = byId("composer-text");
  const status = byId("composer-status");
  const text = textBox.value.trim();
  if (!text) return;
  status.textContent = "Triaging...";
  try {
    const response = await postJson("/v1/triage", { message: text });
    state.testCount += 1;
    const message = { message_id: `test-${state.testCount}`, sender: `Test message ${state.testCount}`, received_at: clockNow(), language: null, text };
    state.messages.push(message);
    state.triage.set(message.message_id, { action: response.action, decision: response.decision });
    state.view = needsAction(message) ? "action" : "done";
    textBox.value = "";
    status.textContent = `Answered in ${Math.round(response.latency_ms)} ms`;
    select(message.message_id);
  } catch (error) {
    status.textContent = `Couldn't triage it: ${error.message}.`;
  }
});

byId("search").addEventListener("input", (event) => {
  state.query = event.target.value.trim();
  render();
});

byId("rerun").addEventListener("click", triageAll);

function setMenuOpen(isOpen) {
  byId("sidebar").classList.toggle("open", isOpen);
  byId("menu-toggle").setAttribute("aria-expanded", String(isOpen));
}

byId("menu-toggle").addEventListener("click", () => setMenuOpen(true));
byId("menu-close").addEventListener("click", () => setMenuOpen(false));

document.addEventListener("keydown", (event) => {
  if (event.key === "/" && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") {
    event.preventDefault();
    byId("search").focus();
  }
});

render();
loadShift();
