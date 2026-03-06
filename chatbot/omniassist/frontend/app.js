/**
 * OmniAssist Frontend
 * Connects to LangGraph Platform API or local LangGraph server.
 */

const API_BASE = "http://localhost:2024";
const GRAPH_NAME = "omniassist";

let threadId = crypto.randomUUID();
let pendingInterrupt = false;

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("user-input");
const agentBadge = document.getElementById("agent-badge");
const threadIdEl = document.getElementById("thread-id");
const approvalBar = document.getElementById("approval-bar");
const approvalPrompt = document.getElementById("approval-prompt");

threadIdEl.textContent = `Thread: ${threadId.slice(0, 8)}`;

inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) sendMessage();
});

function addMessage(content, role, agent = "") {
    const div = document.createElement("div");
    div.className = `message ${role}`;
    if (agent && role === "ai") {
        const tag = document.createElement("div");
        tag.className = "agent-tag";
        tag.textContent = `◆ ${agent}`;
        div.appendChild(tag);
    }
    const text = document.createElement("div");
    text.textContent = content;
    div.appendChild(text);
    messagesEl.appendChild(div);
    messagesEl.parentElement.scrollTop = messagesEl.parentElement.scrollHeight;
    return div;
}

function addThinkingIndicator() {
    const div = document.createElement("div");
    div.className = "message thinking";
    div.id = "thinking-indicator";
    div.textContent = "⚡ Groq is thinking...";
    messagesEl.appendChild(div);
    messagesEl.parentElement.scrollTop = messagesEl.parentElement.scrollHeight;
}

function removeThinkingIndicator() {
    const el = document.getElementById("thinking-indicator");
    if (el) el.remove();
}

async function sendMessage() {
    const content = inputEl.value.trim();
    if (!content) return;

    inputEl.value = "";
    addMessage(content, "human");
    agentBadge.textContent = "Thinking...";
    addThinkingIndicator();

    try {
        const response = await fetch(`${API_BASE}/threads/${threadId}/runs`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                assistant_id: GRAPH_NAME,
                input: { messages: [{ role: "human", content }] },
                stream_mode: ["values"],
            }),
        });

        removeThinkingIndicator();

        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);

        const data = await response.json();
        const messages = data.messages || [];
        const lastAI = messages.filter(m => m.type === "ai" || m.role === "assistant").pop();

        if (lastAI) {
            const agent = data.current_agent || "assistant";
            agentBadge.textContent = agent;
            addMessage(lastAI.content, "ai", agent);
        }

        // Check for interrupt
        if (data.__interrupt__) {
            const interruptData = data.__interrupt__[0] || {};
            showApprovalBar(interruptData.prompt || "This action requires your approval.");
        }
    } catch (error) {
        removeThinkingIndicator();
        addMessage(`❌ Error: ${error.message}`, "ai", "error");
        agentBadge.textContent = "Error";
    }
}

function newThread() {
    threadId = crypto.randomUUID();
    threadIdEl.textContent = `Thread: ${threadId.slice(0, 8)}`;
    messagesEl.innerHTML = "";
    agentBadge.textContent = "Ready";
    approvalBar.style.display = "none";
    addMessage("🔄 New conversation started.", "ai", "system");
}

async function resumeWith(value) {
    approvalBar.style.display = "none";
    agentBadge.textContent = "Resuming...";
    addThinkingIndicator();

    try {
        const response = await fetch(`${API_BASE}/threads/${threadId}/runs`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                assistant_id: GRAPH_NAME,
                command: { resume: value },
            }),
        });

        removeThinkingIndicator();

        const data = await response.json();
        const messages = data.messages || [];
        const lastAI = messages.filter(m => m.type === "ai" || m.role === "assistant").pop();

        if (lastAI) {
            addMessage(lastAI.content, "ai", data.current_agent || "assistant");
        }
        agentBadge.textContent = data.current_agent || "Ready";
    } catch (error) {
        removeThinkingIndicator();
        addMessage(`Resume error: ${error.message}`, "ai", "error");
    }
}

function showApprovalBar(prompt) {
    approvalPrompt.textContent = prompt;
    approvalBar.style.display = "flex";
}
