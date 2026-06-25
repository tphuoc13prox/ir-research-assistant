const topicForm = document.querySelector("#topic-form");
const topicInput = document.querySelector("#topic");
const maxPapersInput = document.querySelector("#max-papers");
const topicSetup = document.querySelector("#topic-setup");
const topicHelp = document.querySelector("#topic-help");
const form = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question");
const topKInput = document.querySelector("#top-k");
const messages = document.querySelector("#messages");
const statusLabel = document.querySelector("#status");

const progressContainer = document.querySelector("#progress-container");
const progressBarFill = document.querySelector("#progress-bar-fill");
const progressStageTitle = document.querySelector("#progress-stage-title");
const progressPercentage = document.querySelector("#progress-percentage");
const progressMessage = document.querySelector("#progress-message");
const consoleLog = document.querySelector("#console-log");

const stageItems = {
  searching: document.querySelector("#stage-searching"),
  downloading: document.querySelector("#stage-downloading"),
  chunking: document.querySelector("#stage-chunking"),
  embedding: document.querySelector("#stage-embedding")
};

let progressTimer = null;
let lastLogMessage = "";

function addConsoleLog(message) {
  if (!message || message === lastLogMessage) return;
  lastLogMessage = message;

  const time = new Date().toLocaleTimeString();
  const entry = document.createElement("div");
  entry.className = "log-entry";
  entry.innerHTML = `<span class="log-time">[${time}]</span> <span class="log-text">${escapeHtml(message)}</span>`;
  consoleLog.appendChild(entry);
  consoleLog.scrollTop = consoleLog.scrollHeight;
}

async function refreshProgress() {
  try {
    const response = await fetch("/session/progress");
    if (!response.ok) return;

    const data = await response.json();
    updateProgressUI(data);

    if (data.stage === "ready") {
      stopProgressPolling();
      await finishProgress(data.topic);
    } else if (data.stage === "error") {
      stopProgressPolling();
      handleProgressError(data.message || "Unknown error during indexing");
    }
  } catch (error) {
    console.error("Progress error:", error);
  }
}

function updateProgressUI(data) {
  const stage = data.stage; // idle, started, searching, search_complete, downloading, chunking, embedding, ready, error
  const message = data.message || "";
  const current = data.current || 0;
  const total = data.total || 0;

  progressMessage.textContent = message;
  statusLabel.textContent = message;

  if (message) {
    addConsoleLog(message);
  }

  // Reset checklist classes
  Object.values(stageItems).forEach(item => {
    item.className = "progress-stage-item pending";
    item.querySelector(".stage-status-text").textContent = "Pending";
  });

  if (stage === "idle" || stage === "started") {
    progressStageTitle.textContent = "Initializing...";
    progressPercentage.textContent = "0%";
    progressBarFill.style.width = "0%";
  } else if (stage === "searching") {
    progressStageTitle.textContent = "Searching arXiv...";
    progressPercentage.textContent = "15%";
    progressBarFill.style.width = "15%";

    stageItems.searching.className = "progress-stage-item active";
    stageItems.searching.querySelector(".stage-status-text").textContent = "Active";
  } else if (stage === "search_complete" || stage === "downloading") {
    progressStageTitle.textContent = "Downloading PDFs...";
    
    const itemPercent = total > 0 ? Math.round((current / total) * 100) : 0;
    const overallPercent = 25 + Math.round(itemPercent * 0.35);
    
    progressPercentage.textContent = `${overallPercent}%`;
    progressBarFill.style.width = `${overallPercent}%`;

    stageItems.searching.className = "progress-stage-item completed";
    stageItems.searching.querySelector(".stage-status-text").textContent = "Done";

    stageItems.downloading.className = "progress-stage-item active";
    stageItems.downloading.querySelector(".stage-status-text").textContent = `${current}/${total}`;
  } else if (stage === "chunking") {
    progressStageTitle.textContent = "Extracting & Chunking...";
    
    const itemPercent = total > 0 ? Math.round((current / total) * 100) : 0;
    const overallPercent = 60 + Math.round(itemPercent * 0.15);
    
    progressPercentage.textContent = `${overallPercent}%`;
    progressBarFill.style.width = `${overallPercent}%`;

    stageItems.searching.className = "progress-stage-item completed";
    stageItems.searching.querySelector(".stage-status-text").textContent = "Done";
    stageItems.downloading.className = "progress-stage-item completed";
    stageItems.downloading.querySelector(".stage-status-text").textContent = "Done";

    stageItems.chunking.className = "progress-stage-item active";
    stageItems.chunking.querySelector(".stage-status-text").textContent = `${current}/${total}`;
  } else if (stage === "embedding") {
    progressStageTitle.textContent = "Indexing Chunks...";
    
    const itemPercent = total > 0 ? Math.round((current / total) * 100) : 0;
    const overallPercent = 75 + Math.round(itemPercent * 0.23);
    
    progressPercentage.textContent = `${overallPercent}%`;
    progressBarFill.style.width = `${overallPercent}%`;

    stageItems.searching.className = "progress-stage-item completed";
    stageItems.searching.querySelector(".stage-status-text").textContent = "Done";
    stageItems.downloading.className = "progress-stage-item completed";
    stageItems.downloading.querySelector(".stage-status-text").textContent = "Done";
    stageItems.chunking.className = "progress-stage-item completed";
    stageItems.chunking.querySelector(".stage-status-text").textContent = "Done";

    stageItems.embedding.className = "progress-stage-item active";
    stageItems.embedding.querySelector(".stage-status-text").textContent = `${current}/${total}`;
  }
}

async function finishProgress(topic) {
  try {
    progressStageTitle.textContent = "Completed!";
    progressPercentage.textContent = "100%";
    progressBarFill.style.width = "100%";

    Object.values(stageItems).forEach(item => {
      item.className = "progress-stage-item completed";
      item.querySelector(".stage-status-text").textContent = "Done";
    });

    addConsoleLog("Pipeline execution completed successfully.");

    const response = await fetch("/session/status");
    if (!response.ok) {
      throw new Error("Failed to load active session details");
    }

    const payload = await response.json();
    
    setTimeout(() => {
      topicSetup.classList.add("is-hidden");
      form.classList.remove("is-hidden");
      messages.innerHTML = "";

      addMessage(
        "assistant",
        `Ready for "${payload.topic}". Indexed ${payload.chunks_indexed} chunks from ${payload.pdfs_downloaded} downloaded PDFs.`
      );

      statusLabel.textContent = "Ready";
      questionInput.focus();
    }, 1200);

  } catch (error) {
    handleProgressError(error.message);
  }
}

function handleProgressError(errorMessage) {
  addMessage("assistant", `Error preparing topic: ${errorMessage}`);
  addConsoleLog(`[ERROR] ${errorMessage}`);
  statusLabel.textContent = "Needs attention";

  topicForm.classList.remove("is-hidden");
  topicHelp.classList.remove("is-hidden");
  progressContainer.classList.add("is-hidden");
  topicForm.querySelector("button").disabled = false;
}

function startProgressPolling() {
  stopProgressPolling();
  refreshProgress();
  progressTimer = setInterval(refreshProgress, 1000);
}

function stopProgressPolling() {
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
}

form.classList.add("is-hidden");

function addMessage(role, text, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  if (sources.length > 0) {
    const sourceList = document.createElement("div");
    sourceList.className = "sources";

    sources.forEach((source, index) => {
      const item = document.createElement("div");
      item.className = "source";

      const preview =
        source.text.length > 260
          ? `${source.text.slice(0, 260)}...`
          : source.text;

      item.innerHTML =
        `<strong>${index + 1}. ${source.paper_id || source.chunk_id}</strong> ` +
        `(score ${source.score.toFixed(3)})<br>${escapeHtml(preview)}`;

      sourceList.appendChild(item);
    });

    bubble.appendChild(sourceList);
  }

  article.appendChild(bubble);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

topicForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const topic = topicInput.value.trim();
  if (!topic) return;

  const maxPapers = Number(maxPapersInput.value || 100);

  statusLabel.textContent = "Preparing topic...";
  topicForm.querySelector("button").disabled = true;

  topicForm.classList.add("is-hidden");
  topicHelp.classList.add("is-hidden");
  progressContainer.classList.remove("is-hidden");

  // Reset the UI progress elements and log console
  progressBarFill.style.width = "0%";
  progressPercentage.textContent = "0%";
  progressStageTitle.textContent = "Initializing...";
  progressMessage.textContent = "Starting request...";
  consoleLog.innerHTML = `<div class="log-entry"><span class="log-time">--:--:--</span> <span class="log-text">Console initialized. Starting pipeline...</span></div>`;
  lastLogMessage = "";

  Object.values(stageItems).forEach(item => {
    item.className = "progress-stage-item pending";
    item.querySelector(".stage-status-text").textContent = "Pending";
  });

  startProgressPolling();

  try {
    const response = await fetch("/session/start", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        topic,
        max_papers: maxPapers,
      }),
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Could not prepare topic");
    }
  } catch (error) {
    stopProgressPolling();
    handleProgressError(error.message);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);

  questionInput.value = "";
  statusLabel.textContent = "Searching";

  // Hide the chat composer to prevent further queries
  form.classList.add("is-hidden");

  // Add assistant loading dots placeholder
  const loadingIndicator = document.createElement("article");
  loadingIndicator.className = "message assistant loading-indicator";
  loadingIndicator.innerHTML = `
    <div class="bubble">
      <span class="loading-dots">Thinking<span>.</span><span>.</span><span>.</span></span>
    </div>
  `;
  messages.appendChild(loadingIndicator);
  messages.scrollTop = messages.scrollHeight;

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        top_k: Number(topKInput.value || 5),
      }),
    });

    const payload = await response.json();

    // Remove loading indicator
    const indicator = document.querySelector(".loading-indicator");
    if (indicator) {
      indicator.remove();
    }

    if (!response.ok) {
      throw new Error(payload.detail || "Request failed");
    }

    addMessage(
      "assistant",
      payload.answer,
      payload.sources
    );

    statusLabel.textContent = "Ready";
  } catch (error) {
    // Remove loading indicator
    const indicator = document.querySelector(".loading-indicator");
    if (indicator) {
      indicator.remove();
    }

    addMessage("assistant", error.message);
    statusLabel.textContent = "Needs attention";
  } finally {
    // Re-enable and show the composer
    form.classList.remove("is-hidden");
    questionInput.focus();
  }
});

// Check active session on page load
async function checkActiveSession() {
  try {
    const progressResponse = await fetch("/session/progress");
    if (progressResponse.ok) {
      const progressData = await progressResponse.json();
      if (progressData.stage && progressData.stage !== "idle" && progressData.stage !== "ready" && progressData.stage !== "error") {
        // A session preparation is already in progress, show progress and poll
        topicForm.classList.add("is-hidden");
        topicHelp.classList.add("is-hidden");
        progressContainer.classList.remove("is-hidden");
        startProgressPolling();
        return;
      }
    }

    const statusResponse = await fetch("/session/status");
    if (statusResponse.ok) {
      const statusData = await statusResponse.json();
      if (statusData.ready) {
        // Active session exists and is completed, hide setup and show chat directly
        topicSetup.classList.add("is-hidden");
        form.classList.remove("is-hidden");
        messages.innerHTML = "";

        addMessage(
          "assistant",
          `Ready for "${statusData.topic}". Indexed ${statusData.chunks_indexed} chunks from ${statusData.pdfs_downloaded} downloaded PDFs.`
        );

        statusLabel.textContent = "Ready";
        questionInput.focus();
      }
    }
  } catch (error) {
    console.error("Error checking active session:", error);
  }
}

// Initialize active session check
checkActiveSession();

