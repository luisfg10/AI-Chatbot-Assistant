// State Management
let selectedModel = "";
let selectedPersonality = "";
let isLoading = false;
let messages = [];

// DOM Elements
const chatPanel = document.getElementById("chatPanel");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const modelSelect = document.getElementById("modelSelect");
const personalitySelect = document.getElementById("personalitySelect");
const clearMemoryButton = document.getElementById("clearMemoryButton");

// Marked.js configuration for markdown rendering
marked.setOptions({
    breaks: true,
    gfm: true,
});

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener("DOMContentLoaded", async () => {
    await initializeModels();
    await initializePersonalities();
    await setDefaultModelAndPersonality();
    setupEventListeners();
});

// ============================================================================
// Initialization Helper
// ============================================================================

async function setDefaultModelAndPersonality() {
    try {
        if (selectedModel) {
            await fetchAPI("/config/model", "POST", { model: selectedModel });
        }
        if (selectedPersonality) {
            await fetchAPI("/config/personality", "POST", { personality: selectedPersonality });
        }
    } catch (error) {
        // Silently fail - defaults will be used by backend
    }
}

// ============================================================================
// API Calls
// ============================================================================

async function fetchAPI(endpoint, method = "GET", body = null) {
    try {
        const options = {
            method,
            headers: {
                "Content-Type": "application/json",
            },
        };

        if (body) {
            options.body = JSON.stringify(body);
        }

        const response = await fetch(`/api${endpoint}`, options);

        if (!response.ok) {
            const text = await response.text();
            try {
                const errorData = JSON.parse(text);
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            } catch {
                throw new Error(`HTTP ${response.status}: ${text.substring(0, 100)}`);
            }
        }

        return await response.json();
    } catch (error) {
        throw new Error(`API Error: ${error.message}`);
    }
}

async function initializeModels() {
    try {
        const data = await fetchAPI("/models");
        selectedModel = data.default_model;
        populateDropdown(modelSelect, data.models, selectedModel);
    } catch (error) {
        showErrorMessage("Failed to load models");
    }
}

async function initializePersonalities() {
    try {
        const data = await fetchAPI("/personalities");
        selectedPersonality = data.default_personality;
        populateDropdown(personalitySelect, data.personalities, selectedPersonality);
    } catch (error) {
        showErrorMessage("Failed to load personalities");
    }
}

async function sendMessage(messageText) {
    if (!messageText.trim() || isLoading) {
        return;
    }

    isLoading = true;
    sendButton.disabled = true;

    const userMessage = messageText.trim();
    messageInput.value = "";
    messageInput.style.height = "auto";

    // Add user message to chat
    addMessage(userMessage, "user");

    // Show typing indicator
    addMessage("", "assistant", true);

    try {
        // Update model and personality if changed
        if (selectedModel !== modelSelect.value) {
            await fetchAPI("/config/model", "POST", {
                model: modelSelect.value,
            });
            selectedModel = modelSelect.value;
        }

        if (selectedPersonality !== personalitySelect.value) {
            await fetchAPI("/config/personality", "POST", {
                personality: personalitySelect.value,
            });
            selectedPersonality = personalitySelect.value;
        }

        // Send message to API
        const response = await fetchAPI("/chat", "POST", {
            message: userMessage,
        });

        // Remove typing indicator and add assistant response
        chatPanel.removeChild(chatPanel.lastChild);
        addMessage(response.response, "assistant");
    } catch (error) {
        // Remove typing indicator
        if (chatPanel.lastChild && chatPanel.lastChild.querySelector(".typing-indicator")) {
            chatPanel.removeChild(chatPanel.lastChild);
        }
        showErrorMessage(error.message);
    } finally {
        isLoading = false;
        sendButton.disabled = false;
        messageInput.focus();
    }
}

async function clearMemory() {
    if (!confirm("Are you sure you want to clear the conversation memory?")) {
        return;
    }

    try {
        await fetchAPI("/reset", "POST");
        messages = [];
        chatPanel.innerHTML = "";
        showSystemMessage("Conversation memory cleared.");
    } catch (error) {
        showErrorMessage("Failed to clear memory");
    }
}

// ============================================================================
// DOM Manipulation
// ============================================================================

function addMessage(text, sender, isTyping = false) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}`;

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    if (isTyping) {
        bubble.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
    } else {
        const contentDiv = document.createElement("div");
        contentDiv.className = "message-content";
        contentDiv.innerHTML = marked.parse(text);
        bubble.appendChild(contentDiv);

        // Add timestamp
        const timeDiv = document.createElement("div");
        timeDiv.className = "message-time";
        timeDiv.textContent = new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
        });
        bubble.appendChild(timeDiv);
    }

    messageDiv.appendChild(bubble);
    chatPanel.appendChild(messageDiv);

    // Auto-scroll to bottom
    chatPanel.scrollTop = chatPanel.scrollHeight;

    if (!isTyping) {
        messages.push({ sender, text, timestamp: new Date() });
    }
}

function showErrorMessage(errorText) {
    const messageDiv = document.createElement("div");
    messageDiv.className = "message error";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    contentDiv.textContent = errorText;

    bubble.appendChild(contentDiv);
    messageDiv.appendChild(bubble);
    chatPanel.appendChild(messageDiv);

    chatPanel.scrollTop = chatPanel.scrollHeight;
}

function showSystemMessage(text) {
    const messageDiv = document.createElement("div");
    messageDiv.className = "message system";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    contentDiv.textContent = text;

    bubble.appendChild(contentDiv);
    messageDiv.appendChild(bubble);
    chatPanel.appendChild(messageDiv);

    chatPanel.scrollTop = chatPanel.scrollHeight;
}

function populateDropdown(selectElement, items, selectedValue) {
    selectElement.innerHTML = "";
    items.forEach((item) => {
        const option = document.createElement("option");
        option.value = item;
        option.textContent = item;
        if (item === selectedValue) {
            option.selected = true;
        }
        selectElement.appendChild(option);
    });
}

// ============================================================================
// Event Listeners
// ============================================================================

function setupEventListeners() {
    // Send button click
    sendButton.addEventListener("click", () => {
        sendMessage(messageInput.value);
    });

    // Enter to send, Shift+Enter for newline
    messageInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage(messageInput.value);
        }
    });

    // Auto-resize textarea
    messageInput.addEventListener("input", () => {
        messageInput.style.height = "auto";
        messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
    });

    // Model selection change
    modelSelect.addEventListener("change", async (event) => {
        const newModel = event.target.value;
        try {
            await fetchAPI("/config/model", "POST", { model: newModel });
            selectedModel = newModel;
        } catch (error) {
            showErrorMessage(`Failed to change model: ${error.message}`);
            modelSelect.value = selectedModel;
        }
    });

    // Personality selection change
    personalitySelect.addEventListener("change", async (event) => {
        const newPersonality = event.target.value;
        try {
            await fetchAPI("/config/personality", "POST", { personality: newPersonality });
            selectedPersonality = newPersonality;
        } catch (error) {
            showErrorMessage(`Failed to change personality: ${error.message}`);
            personalitySelect.value = selectedPersonality;
        }
    });

    // Clear memory button
    clearMemoryButton.addEventListener("click", clearMemory);
}
