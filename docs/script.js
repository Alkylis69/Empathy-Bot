/* global EMPATHY_BOT_CONFIG */
const state = {
  endpoint: localStorage.getItem("empathy.endpoint") || EMPATHY_BOT_CONFIG.defaultEndpoint,
  simulateStream: localStorage.getItem("empathy.simStream") === "true",
  theme: localStorage.getItem("empathy.theme") || "dark",
  messages: loadMessages()
};
const dom = {
  chat: document.getElementById("chat"),
  form: document.getElementById("chat-form"),
  input: document.getElementById("user-input"),
  sendBtn: document.getElementById("send-btn"),
  clearChat: document.getElementById("clear-chat"),
  toggleTheme: document.getElementById("toggle-theme"),
  openSettings: document.getElementById("open-settings"),
  settingsDialog: document.getElementById("settings-dialog"),
  apiEndpoint: document.getElementById("api-endpoint"),
  simulateStream: document.getElementById("simulate-stream"),
  saveSettings: document.getElementById("save-settings"),
  template: document.getElementById("message-template")
};
// Initialization
applyTheme(state.theme);
renderAll();
scrollToBottom();
dom.form.addEventListener("submit", handleSubmit);
dom.clearChat.addEventListener("click", clearConversation);
dom.toggleTheme.addEventListener("click", toggleTheme);
dom.openSettings.addEventListener("click", openSettings);
dom.saveSettings.addEventListener("click", saveSettings);
// Auto-resize textarea
dom.input.addEventListener("input", () => {
  dom.input.style.height = "auto";
  dom.input.style.height = dom.input.scrollHeight + "px";
});
function loadMessages() {
  try {
    const raw = localStorage.getItem("empathy.messages");
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}
function persist() {
  localStorage.setItem("empathy.messages", JSON.stringify(state.messages));
}
function uuid() {
  return crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2);
}
function addMessage(role, content, extra = {}) {
  const msg = { id: uuid(), role, content, ts: new Date().toISOString(), ...extra };
  state.messages.push(msg);
  persist();
  renderMessage(msg);
  return msg;
}
function renderAll() {
  dom.chat.innerHTML = "";
  state.messages.forEach(renderMessage);
}
function renderMessage(msg) {
  const clone = dom.template.content.firstElementChild.cloneNode(true);
  clone.classList.add(msg.role);
  clone.querySelector(".role").textContent = msg.role === "user" ? "You" : (msg.role === "assistant" ? "Empathy Bot" : msg.role);
  clone.querySelector(".timestamp").textContent = formatTime(msg.ts);
  const contentEl = clone.querySelector(".content");
  if (msg.error) clone.classList.add("error");
  contentEl.innerHTML = renderMarkdownBasic(msg.content);
  dom.chat.appendChild(clone);
  requestAnimationFrame(scrollToBottom);
}
function updateMessage(id, partial) {
  const idx = state.messages.findIndex(m => m.id === id);
  if (idx === -1) return;
  state.messages[idx] = { ...state.messages[idx], ...partial };
  persist();
  // Simpler: re-render last if it's the last; else full re-render
  if (idx === state.messages.length - 1) {
    dom.chat.lastElementChild?.remove();
    renderMessage(state.messages[idx]);
  } else {
    renderAll();
  }
}
function formatTime(ts) {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
async function handleSubmit(e) {
  e.preventDefault();
  const text = dom.input.value.trim();
  if (!text) return;
  dom.sendBtn.disabled = true;
  addMessage("user", text);
  dom.input.value = "";
  dom.input.style.height = "auto";
  const assistantMsg = addMessage("assistant", "Thinking...", { pending: true });
  try {
    const reply = await queryBackend(text, assistantMsg.id);
    if (!reply) throw new Error("Empty response");
    updateMessage(assistantMsg.id, { content: reply, pending: false });
  } catch (err) {
    updateMessage(assistantMsg.id, {
      content: "An error occurred: " + (err.message || err),
      pending: false,
      error: true
    });
    console.error(err);
  } finally {
    dom.sendBtn.disabled = false;
  }
}
async function queryBackend(userText, tempMsgId) {
  if (!state.endpoint) throw new Error("API endpoint not set. Open Settings (⚙️) to configure.");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), EMPATHY_BOT_CONFIG.requestTimeoutMs);
  const payload = { message: userText };
  const res = await fetch(state.endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: controller.signal
  }).finally(() => clearTimeout(timeout));
  if (!res.ok) throw new Error("HTTP " + res.status);
  const data = await res.json();
  const reply = data.reply || data.response || JSON.stringify(data);
  if (state.simulateStream) {
    await simulateStreaming(tempMsgId, reply);
    return reply;
  }
  return reply;
}
async function simulateStreaming(id, fullText) {
  const tokens = fullText.split(/(\\s+)/);
  let acc = "";
  for (let i = 0; i < tokens.length; i++) {
    acc += tokens[i];
    updateMessage(id, { content: acc + "▌" });
    await delay(35 + Math.random() * 60);
  }
  updateMessage(id, { content: acc });
}
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }
function clearConversation() {
  if (!confirm("Clear all messages?")) return;
  state.messages = [];
  persist();
  renderAll();
}
function toggleTheme() {
  state.theme = state.theme === "dark" ? "light" : "dark";
  localStorage.setItem("empathy.theme", state.theme);
  applyTheme(state.theme);
}
function applyTheme(theme) {
  if (theme === "light") document.documentElement.classList.add("light");
  else document.documentElement.classList.remove("light");
}
function openSettings() {
  dom.apiEndpoint.value = state.endpoint;
  dom.simulateStream.checked = state.simulateStream;
  dom.settingsDialog.showModal();
}
function saveSettings(ev) {
  ev.preventDefault();
  const ep = dom.apiEndpoint.value.trim();
  if (ep) {
    state.endpoint = ep;
    localStorage.setItem("empathy.endpoint", ep);
  }
  state.simulateStream = dom.simulateStream.checked;
  localStorage.setItem("empathy.simStream", state.simulateStream);
  dom.settingsDialog.close();
}
function scrollToBottom() {
  dom.chat.scrollTop = dom.chat.scrollHeight;
}
// Minimal markdown-ish rendering
function renderMarkdownBasic(text) {
  const esc = s => s.replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  let html = esc(text);
  html = html
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>")
    .replace(/\\*([^*]+)\\*/g, "<em>$1</em>")
    .replace(/\\n{2,}/g, "</p><p>")
    .replace(/\\n/g, "<br/>");
  return "<p>" + html + "</p>";
}