// Global State
let discoveredPapers = [];
let selectedPapers = [];
let activePaperId = null;
let activePaperTitle = null;
let progressTimer = null;
let lastLogMessage = "";

// DOM Elements - Screen Containers
const topicSetupScreen = document.querySelector("#topic-setup-screen");
const kbBuilderScreen = document.querySelector("#kb-builder-screen");
const chatWorkspaceScreen = document.querySelector("#chat-workspace-screen");

// DOM Elements - Topic Setup & Search Selection
const topicForm = document.querySelector("#topic-form");
const topicInput = document.querySelector("#topic");
const baseModelSelect = document.querySelector("#base-model");
const searchBtn = document.querySelector("#search-btn");
const searchResultsContainer = document.querySelector("#search-results-container");
const searchResultsBody = document.querySelector("#search-results-body");
const resultsCountLabel = document.querySelector("#results-count");
const selectAllBtn = document.querySelector("#select-all-btn");
const clearAllBtn = document.querySelector("#clear-all-btn");
const importBtn = document.querySelector("#import-btn");

// DOM Elements - KB Builder Progress Screen
const progressStageTitle = document.querySelector("#progress-stage-title");
const progressPercentage = document.querySelector("#progress-percentage");
const progressBarFill = document.querySelector("#progress-bar-fill");
const progressMessage = document.querySelector("#progress-message");
const consoleLogStream = document.querySelector("#console-log");
const summaryCard = document.querySelector("#summary-card");
const launchChatBtn = document.querySelector("#start-chat-btn");

const progressStages = {
  downloading: document.querySelector("#stage-downloading"),
  parsing: document.querySelector("#stage-parsing"),
  chunking: document.querySelector("#stage-chunking"),
  embedding: document.querySelector("#stage-embedding")
};

// DOM Elements - Chat Workspace & Sidebar
const papersSidebar = document.querySelector(".papers-sidebar");
const papersCountBadge = document.querySelector("#papers-count-badge");
const papersList = document.querySelector("#papers-list");
const chatForm = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question");
const topKInput = document.querySelector("#top-k");
const messagesContainer = document.querySelector("#messages");
const statusDot = document.querySelector("#status-dot");
const statusLabel = document.querySelector("#status");
const activeScopeBadge = document.querySelector("#active-scope-badge");

// DOM Elements - Right Sidebar Panels
const tabButtons = document.querySelectorAll(".tab-btn");
const tabPanels = document.querySelectorAll(".tab-panel");
const settingsForm = document.querySelector("#settings-form");

// DOM Elements - PDF Viewer Sidebar
const workspaceInfoSidebar = document.querySelector("#workspace-info-sidebar");
const pdfViewerSidebar = document.querySelector("#pdf-viewer-sidebar");
const pdfViewerTitle = document.querySelector("#pdf-viewer-title");
const pdfViewerIframe = document.querySelector("#pdf-viewer-iframe");
const closePdfBtn = document.querySelector("#close-pdf-btn");

// Right Sidebar - Stats Panel UI
const statsPapers = document.querySelector("#stats-papers");
const statsChunks = document.querySelector("#stats-chunks");
const statsModel = document.querySelector("#stats-model");
const statsHybrid = document.querySelector("#stats-hybrid");
const statsDense = document.querySelector("#stats-dense");
const statsSparse = document.querySelector("#stats-sparse");

// Right Sidebar - Debug Panel UI
const dbgDenseTime = document.querySelector("#dbg-dense-time");
const dbgSparseTime = document.querySelector("#dbg-sparse-time");
const dbgFusionTime = document.querySelector("#dbg-fusion-time");
const dbgTotalTime = document.querySelector("#dbg-total-time");
const rankToggleButtons = document.querySelectorAll(".rank-toggle-btn");
const rankListPanels = document.querySelectorAll(".ranks-list-panel");
const fusedRanksBody = document.querySelector("#fused-ranks-body");
const denseRanksBody = document.querySelector("#dense-ranks-body");
const sparseRanksBody = document.querySelector("#sparse-ranks-body");

// ------------------------------------------------------------------
// UTILITIES
// ------------------------------------------------------------------
function escapeHtml(value) {
  if (!value) return "";
  return value
    .toString()
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatMarkdown(text) {
  return escapeHtml(text)
    .replaceAll(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replaceAll(/^### (.*?)$/gm, "<h3>$1</h3>")
    .replaceAll("\n", "<br>");
}

function addConsoleLog(message) {
  if (!message || message === lastLogMessage) return;
  lastLogMessage = message;
  
  const entry = document.createElement("div");
  entry.className = "log-entry";
  
  const timestamp = new Date().toLocaleTimeString();
  entry.innerHTML = `<span class="log-time">[${timestamp}]</span> <span class="log-text">${escapeHtml(message)}</span>`;
  
  consoleLogStream.appendChild(entry);
  consoleLogStream.scrollTop = consoleLogStream.scrollHeight;
}

function updateStatus(dotClass, labelText) {
  statusDot.className = `status-dot ${dotClass}`;
  statusLabel.textContent = labelText;
}

// ------------------------------------------------------------------
// DISCOVERY & SEARCH SCREEN ACTIONS
// ------------------------------------------------------------------
topicForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = topicInput.value.trim();
  if (!query) return;

  searchBtn.disabled = true;
  searchBtn.textContent = "Discovering...";
  updateStatus("active", "Searching papers...");

  try {
    const response = await fetch(`/session/search?query=${encodeURIComponent(query)}&limit=15`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not retrieve papers metadata");
    }

    discoveredPapers = payload;
    renderSearchResults(discoveredPapers);
    updateStatus("idle", "Idle");
  } catch (error) {
    alert(`Search error: ${error.message}`);
    updateStatus("error", "Error");
  } finally {
    searchBtn.disabled = false;
    searchBtn.textContent = "Discover Papers";
  }
});

function renderSearchResults(papers) {
  searchResultsContainer.classList.remove("is-hidden");
  searchResultsBody.innerHTML = "";
  resultsCountLabel.textContent = papers.length;
  
  selectedPapers = [];
  importBtn.disabled = true;

  if (papers.length === 0) {
    searchResultsBody.innerHTML = `<tr><td colspan="4" class="empty-row">No papers discovered matching that topic. Try another query.</td></tr>`;
    return;
  }

  papers.forEach(paper => {
    const tr = document.createElement("tr");
    tr.dataset.id = paper.paper_id;
    
    const authorsList = Array.isArray(paper.authors) ? paper.authors.join(", ") : paper.authors;
    const abstractText = paper.abstract || "No abstract details available.";
    const categoriesTags = paper.categories.map(c => `<span class="category-tag">${escapeHtml(c)}</span>`).join("");

    tr.innerHTML = `
      <td class="col-check">
        <input type="checkbox" class="paper-select-check" data-id="${escapeHtml(paper.paper_id)}" />
      </td>
      <td>
        <div class="col-title">${escapeHtml(paper.title)}</div>
        <div class="col-authors">${escapeHtml(authorsList)}</div>
        <div class="paper-abstract-desc">${escapeHtml(abstractText)}</div>
      </td>
      <td class="col-year">${escapeHtml(paper.published_year)}</td>
      <td class="col-category">${categoriesTags}</td>
    `;

    // Row selection toggle click handler
    tr.querySelector(".paper-select-check").addEventListener("change", (e) => {
      const checkbox = e.target;
      if (checkbox.checked) {
        tr.classList.add("selected");
        selectedPapers.push(paper);
      } else {
        tr.classList.remove("selected");
        selectedPapers = selectedPapers.filter(p => p.paper_id !== paper.paper_id);
      }
      importBtn.disabled = selectedPapers.length === 0;
    });

    searchResultsBody.appendChild(tr);
  });
}

// Select All action
selectAllBtn.addEventListener("click", () => {
  document.querySelectorAll(".paper-select-check").forEach(checkbox => {
    if (!checkbox.checked) {
      checkbox.checked = true;
      checkbox.dispatchEvent(new Event("change"));
    }
  });
});

// Clear Selection action
clearAllBtn.addEventListener("click", () => {
  document.querySelectorAll(".paper-select-check").forEach(checkbox => {
    if (checkbox.checked) {
      checkbox.checked = false;
      checkbox.dispatchEvent(new Event("change"));
    }
  });
});

// Ingestion import submit action
importBtn.addEventListener("click", async () => {
  if (selectedPapers.length === 0) return;
  const topic = topicInput.value.trim();
  const baseModel = baseModelSelect.value;

  importBtn.disabled = true;
  updateStatus("downloading", "Preparing Builder...");

  topicSetupScreen.classList.add("is-hidden");
  kbBuilderScreen.classList.remove("is-hidden");
  
  progressBarFill.style.width = "0%";
  progressPercentage.textContent = "0%";
  progressMessage.textContent = "Starting request...";
  consoleLogStream.innerHTML = "";
  summaryCard.classList.add("is-hidden");

  // Reset checklist stages
  Object.values(progressStages).forEach(stage => {
    stage.className = "progress-stage-item pending";
    stage.querySelector(".stage-status-text").textContent = "Pending";
  });

  try {
    const response = await fetch("/session/start", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        topic,
        papers: selectedPapers,
        base_model_name: baseModel,
      }),
    });

    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "Build request failed");
    }

    startProgressPolling();
  } catch (error) {
    updateStatus("error", "Error");
    alert(`Could not start build pipeline: ${error.message}`);
  }
});

// ------------------------------------------------------------------
// PROGRESS CHECKLIST POLLING (KB BUILDER)
// ------------------------------------------------------------------
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

async function refreshProgress() {
  try {
    const response = await fetch("/session/progress");
    if (!response.ok) return;

    const data = await response.json();
    updateProgressUI(data);

    if (data.stage === "ready") {
      stopProgressPolling();
      handleProgressComplete(data.summary);
    } else if (data.stage === "error") {
      stopProgressPolling();
      updateStatus("error", "Error");
      progressMessage.textContent = data.message || "Build failed.";
      addConsoleLog(`[ERROR] Build pipeline failed: ${data.message}`);
    }
  } catch (error) {
    console.error("Error polling builder progress:", error);
  }
}

function updateProgressUI(data) {
  const stage = data.stage;
  const message = data.message || "";
  const current = data.current || 0;
  const total = data.total || 0;

  if (message) {
    addConsoleLog(message);
    progressMessage.textContent = message;
  }

  // Reset all stages to pending by default
  Object.values(progressStages).forEach(item => {
    item.className = "progress-stage-item pending";
    item.querySelector(".stage-status-text").textContent = "Pending";
  });

  if (stage === "started") {
    progressStageTitle.textContent = "Initializing stages...";
    progressPercentage.textContent = "0%";
    progressBarFill.style.width = "0%";
    updateStatus("active", "Initializing...");
  }
  else if (stage === "downloading") {
    progressStageTitle.textContent = "Downloading PDF Files...";
    updateStatus("downloading", "Downloading PDFs...");
    const pct = total > 0 ? Math.round((current / total) * 100) : 0;
    const overall = 5 + Math.round(pct * 0.35);
    
    progressPercentage.textContent = `${overall}%`;
    progressBarFill.style.width = `${overall}%`;

    progressStages.downloading.className = "progress-stage-item active";
    progressStages.downloading.querySelector(".stage-status-text").textContent = `${current}/${total}`;
  }
  else if (stage === "parsing") {
    progressStageTitle.textContent = "Parsing PDF structures...";
    updateStatus("active", "Parsing PDF text...");
    const pct = total > 0 ? Math.round((current / total) * 100) : 0;
    const overall = 40 + Math.round(pct * 0.20);
    
    progressPercentage.textContent = `${overall}%`;
    progressBarFill.style.width = `${overall}%`;

    progressStages.downloading.className = "progress-stage-item completed";
    progressStages.downloading.querySelector(".stage-status-text").textContent = "Done";

    progressStages.parsing.className = "progress-stage-item active";
    progressStages.parsing.querySelector(".stage-status-text").textContent = `${current}/${total}`;
  }
  else if (stage === "chunking") {
    progressStageTitle.textContent = "Splitting text into chunks...";
    updateStatus("active", "Chunking text...");
    const pct = total > 0 ? Math.round((current / total) * 100) : 0;
    const overall = 60 + Math.round(pct * 0.15);
    
    progressPercentage.textContent = `${overall}%`;
    progressBarFill.style.width = `${overall}%`;

    progressStages.downloading.className = "progress-stage-item completed";
    progressStages.downloading.querySelector(".stage-status-text").textContent = "Done";
    progressStages.parsing.className = "progress-stage-item completed";
    progressStages.parsing.querySelector(".stage-status-text").textContent = "Done";

    progressStages.chunking.className = "progress-stage-item active";
    progressStages.chunking.querySelector(".stage-status-text").textContent = `${current}/${total}`;
  }
  else if (stage === "embedding") {
    progressStageTitle.textContent = "Indexing vector keys...";
    updateStatus("indexing", "Generating Index...");
    const pct = total > 0 ? Math.round((current / total) * 100) : 0;
    const overall = 75 + Math.round(pct * 0.23);
    
    progressPercentage.textContent = `${overall}%`;
    progressBarFill.style.width = `${overall}%`;

    progressStages.downloading.className = "progress-stage-item completed";
    progressStages.downloading.querySelector(".stage-status-text").textContent = "Done";
    progressStages.parsing.className = "progress-stage-item completed";
    progressStages.parsing.querySelector(".stage-status-text").textContent = "Done";
    progressStages.chunking.className = "progress-stage-item completed";
    progressStages.chunking.querySelector(".stage-status-text").textContent = "Done";

    progressStages.embedding.className = "progress-stage-item active";
    progressStages.embedding.querySelector(".stage-status-text").textContent = `${current}/${total}`;
  }
}

function handleProgressComplete(summary) {
  progressStageTitle.textContent = "KB Build Completed!";
  progressPercentage.textContent = "100%";
  progressBarFill.style.width = "100%";
  progressMessage.textContent = "Indices built. Review the summary card to proceed.";
  updateStatus("idle", "Build Success");

  Object.values(progressStages).forEach(stage => {
    stage.className = "progress-stage-item completed";
    stage.querySelector(".stage-status-text").textContent = "Done";
  });

  addConsoleLog("Hybrid indexing completed successfully.");

  // Hydrate summary card details
  if (summary) {
    document.querySelector("#sum-papers").textContent = summary.papers_imported;
    document.querySelector("#sum-chunks").textContent = summary.chunks_generated;
    document.querySelector("#sum-emb-time").textContent = `${summary.embedding_time_seconds}s`;
    
    // Format index size to readable KB/MB
    const sizeBytes = summary.index_size_bytes || 0;
    const sizeMB = sizeBytes / (1024 * 1024);
    if (sizeMB >= 1.0) {
      document.querySelector("#sum-index-size").textContent = `${sizeMB.toFixed(2)} MB`;
    } else {
      document.querySelector("#sum-index-size").textContent = `${(sizeBytes / 1024).toFixed(1)} KB`;
    }
  }

  summaryCard.classList.remove("is-hidden");
}

// Click Launch Chat workspace button
launchChatBtn.addEventListener("click", async () => {
  kbBuilderScreen.classList.add("is-hidden");
  chatWorkspaceScreen.classList.remove("is-hidden");
  
  // Hydrate active chat session metadata
  await loadWorkspaceDetails();
});

// ------------------------------------------------------------------
// CHAT WORKSPACE & SIDEBAR DETAILS
// ------------------------------------------------------------------
async function loadWorkspaceDetails() {
  try {
    const response = await fetch("/session/status");
    if (!response.ok) return;

    const payload = await response.json();
    if (payload.ready) {
      messagesContainer.innerHTML = "";
      addMessage(
        "assistant",
        `Ready for topic "${payload.topic}". Ingested ${payload.chunks_indexed} chunks from ${payload.pdfs_downloaded} papers.`
      );

      renderPapersSidebar(payload.papers);
      hydrateStatsTab(payload.stats);
      await loadSettingsTab();
      updateStatus("idle", "Ready");
    }
  } catch (error) {
    console.error("Could not load active workspace details:", error);
  }
}

function renderPapersSidebar(papers) {
  papersCountBadge.textContent = papers.length;
  papersList.innerHTML = "";
  
  if (papers.length === 0) {
    papersList.innerHTML = `<p class="empty-row">No ingested papers.</p>`;
    return;
  }

  papers.forEach(paper => {
    const authorsText = Array.isArray(paper.authors) ? paper.authors.join(", ") : paper.authors;
    const item = document.createElement("div");
    item.className = "paper-item";
    item.dataset.paperId = paper.paper_id;
    item.dataset.paperTitle = paper.title;

    item.innerHTML = `
      <span class="paper-item-status-badge"></span>
      <h3 class="paper-title">${escapeHtml(paper.title)}</h3>
      <p class="paper-authors">${escapeHtml(authorsText)}</p>
    `;

    item.addEventListener("click", () => handlePaperClick(item, paper.paper_id, paper.title));
    papersList.appendChild(item);
  });

  // Global Scope Reset badge click
  const globalReset = document.createElement("div");
  globalReset.className = "paper-item active";
  globalReset.style.marginTop = "auto";
  globalReset.style.borderStyle = "dashed";
  globalReset.innerHTML = `
    <h3 class="paper-title" style="text-align: center; color: var(--accent);">🌐 Reset to Global Scope</h3>
  `;
  globalReset.addEventListener("click", () => {
    document.querySelectorAll(".paper-item").forEach(el => el.classList.remove("active"));
    globalReset.classList.add("active");
    
    activePaperId = null;
    activePaperTitle = null;
    activeScopeBadge.className = "scope-badge global";
    activeScopeBadge.textContent = "Global Scope";
    questionInput.placeholder = "Ask a question about all papers...";

    // Reset PDF Viewer and show Stats panel
    pdfViewerIframe.src = "";
    pdfViewerSidebar.classList.add("is-hidden");
    workspaceInfoSidebar.classList.remove("is-hidden");
  });
  papersList.appendChild(globalReset);
}

async function handlePaperClick(item, paperId, paperTitle) {
  document.querySelectorAll(".paper-item").forEach(el => el.classList.remove("active"));
  item.classList.add("active");

  activePaperId = paperId;
  activePaperTitle = paperTitle;

  activeScopeBadge.className = "scope-badge paper";
  const displayTitle = paperTitle.length > 30 ? `${paperTitle.slice(0, 30)}...` : paperTitle;
  activeScopeBadge.textContent = displayTitle;

  questionInput.placeholder = `Ask a question about "${displayTitle}"...`;
  questionInput.focus();

  // Load original PDF in right-side sidebar iframe
  pdfViewerTitle.textContent = paperTitle;
  pdfViewerIframe.src = `/paper/${paperId}/pdf`;
  
  // Show PDF viewer panel and hide Stats panel
  workspaceInfoSidebar.classList.add("is-hidden");
  pdfViewerSidebar.classList.remove("is-hidden");
}

function addMessage(role, text, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = formatMarkdown(text);

  if (sources.length > 0) {
    const details = document.createElement("details");
    details.className = "sources-accordion";

    const summary = document.createElement("summary");
    summary.textContent = `Show Citations (${sources.length} sources)`;
    details.appendChild(summary);

    const sourceList = document.createElement("div");
    sourceList.className = "sources";

    sources.forEach((source, index) => {
      const item = document.createElement("div");
      item.className = "source";
      const preview = source.text.length > 250 ? `${source.text.slice(0, 250)}...` : source.text;
      
      item.innerHTML = `
        <strong>${index + 1}. Chunk ID: ${escapeHtml(source.chunk_id)}</strong> 
        (Score: ${source.score.toFixed(3)})<br>${escapeHtml(preview)}
      `;
      sourceList.appendChild(item);
    });

    details.appendChild(sourceList);
    bubble.appendChild(details);
  }

  article.appendChild(bubble);
  messagesContainer.appendChild(article);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Composer submit chat message action
chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";
  updateStatus("active", "Generating answer...");

  // Add assistant typing bubble
  const loadingIndicator = document.createElement("article");
  loadingIndicator.className = "message assistant loading-indicator";
  loadingIndicator.innerHTML = `
    <div class="bubble">
      <span class="loading-dots">Assistant is thinking<span>.</span><span>.</span><span>.</span></span>
    </div>
  `;
  messagesContainer.appendChild(loadingIndicator);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        top_k: Number(topKInput.value || 5),
        paper_id: activePaperId,
      }),
    });

    const payload = await response.json();
    
    const indicator = document.querySelector(".loading-indicator");
    if (indicator) indicator.remove();

    if (!response.ok) {
      throw new Error(payload.detail || "Server communication error");
    }

    addMessage("assistant", payload.answer, payload.sources);
    updateStatus("idle", "Ready");

    // Populate Debug telemetry items
    if (payload.debug_info) {
      renderDebugTelemetry(payload.debug_info);
    }
  } catch (error) {
    const indicator = document.querySelector(".loading-indicator");
    if (indicator) indicator.remove();

    addMessage("assistant", `An error occurred: ${error.message}`);
    updateStatus("error", "Error");
  } finally {
    questionInput.focus();
  }
});

// ------------------------------------------------------------------
// TAB NAVIGATION & PANELS INTERACTION
// ------------------------------------------------------------------
tabButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    tabButtons.forEach(b => b.classList.remove("active"));
    tabPanels.forEach(p => p.classList.remove("active"));

    btn.classList.add("active");
    const tabId = btn.dataset.tab;
    document.getElementById(tabId).classList.add("active");

    // Live update triggers
    if (tabId === "tab-stats") {
      updateStatsPanel();
    } else if (tabId === "tab-settings") {
      loadSettingsTab();
    }
  });
});

// 📊 Hydrate Stats Panel
function hydrateStatsTab(stats) {
  if (!stats) return;
  statsPapers.textContent = stats.papers_count;
  statsChunks.textContent = stats.chunks_count;
  statsModel.textContent = stats.embedding_model;
  statsHybrid.textContent = stats.hybrid_enabled ? "Enabled (RRF)" : "Disabled (Dense Only)";
  
  statsDense.className = `stats-value status-indicator ${stats.dense_index_status === "built" ? "built" : "none"}`;
  statsDense.textContent = stats.dense_index_status === "built" ? "Built (FAISS)" : "None";

  statsSparse.className = `stats-value status-indicator ${stats.sparse_index_status === "built" ? "built" : "none"}`;
  statsSparse.textContent = stats.sparse_index_status === "built" ? "Built (BM25)" : "None";
}

async function updateStatsPanel() {
  try {
    const response = await fetch("/session/status");
    if (response.ok) {
      const payload = await response.json();
      if (payload.ready) {
        hydrateStatsTab(payload.stats);
      }
    }
  } catch (e) {
    console.error("Could not fetch active status for stats tab:", e);
  }
}

// ⚙️ Load Settings Tab Form values
async function loadSettingsTab() {
  try {
    const response = await fetch("/settings");
    if (!response.ok) return;

    const payload = await response.json();
    const sets = payload.settings;
    
    document.querySelector("#setting-hybrid-enabled").checked = sets.hybrid_enabled;
    document.querySelector("#setting-dense-k").value = sets.dense_top_k;
    document.querySelector("#setting-sparse-k").value = sets.sparse_top_k;
    document.querySelector("#setting-fusion-k").value = sets.fusion_top_k;
    document.querySelector("#setting-rrf-k").value = sets.rrf_k;
    document.querySelector("#setting-dense-w").value = sets.dense_weight !== undefined ? sets.dense_weight : 1.0;
    document.querySelector("#setting-sparse-w").value = sets.sparse_weight !== undefined ? sets.sparse_weight : 1.0;
    document.querySelector("#setting-ranking-enabled").checked = sets.ranking_enabled !== undefined ? sets.ranking_enabled : true;
    document.querySelector("#setting-base-model").value = sets.base_model_name !== undefined ? sets.base_model_name : "Qwen/Qwen2.5-0.5B-Instruct";
    document.querySelector("#setting-relevance-threshold").value = sets.relevance_threshold !== undefined ? sets.relevance_threshold : 0.35;
  } catch (error) {
    console.error("Error loading settings form values:", error);
  }
}

// Submit Settings update form
settingsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  
  const payload = {
    hybrid_enabled: document.querySelector("#setting-hybrid-enabled").checked,
    dense_top_k: Number(document.querySelector("#setting-dense-k").value),
    sparse_top_k: Number(document.querySelector("#setting-sparse-k").value),
    fusion_top_k: Number(document.querySelector("#setting-fusion-k").value),
    rrf_k: Number(document.querySelector("#setting-rrf-k").value),
    dense_weight: Number(document.querySelector("#setting-dense-w").value),
    sparse_weight: Number(document.querySelector("#setting-sparse-w").value),
    ranking_enabled: document.querySelector("#setting-ranking-enabled").checked,
    base_model_name: document.querySelector("#setting-base-model").value,
    relevance_threshold: Number(document.querySelector("#setting-relevance-threshold").value),
  };

  try {
    const response = await fetch("/settings", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (response.ok) {
      alert("Retrieval settings updated successfully.");
      await updateStatsPanel();
    } else {
      alert("Failed to update retrieval configurations.");
    }
  } catch (error) {
    alert(`Settings update error: ${error.message}`);
  }
});

// 🛠️ Debug Ranks Panels View toggling
rankToggleButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    rankToggleButtons.forEach(b => b.classList.remove("active"));
    rankListPanels.forEach(p => p.classList.remove("active"));

    btn.classList.add("active");
    const panelId = btn.dataset.rankView;
    document.getElementById(panelId).classList.add("active");
  });
});

// Render debug query telemetry inside the developer view
function renderDebugTelemetry(debug) {
  dbgDenseTime.textContent = `${debug.dense_latency_ms.toFixed(1)} ms`;
  dbgSparseTime.textContent = `${debug.sparse_latency_ms.toFixed(1)} ms`;
  dbgFusionTime.textContent = `${debug.fusion_latency_ms.toFixed(1)} ms`;
  dbgTotalTime.textContent = `${debug.total_latency_ms.toFixed(1)} ms`;

  // Render Fused Ranks
  fusedRanksBody.innerHTML = "";
  if (debug.fused_results && debug.fused_results.length > 0) {
    debug.fused_results.forEach((item, index) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${index + 1}</td>
        <td>${escapeHtml(item.chunk_id)}</td>
        <td>${item.score.toFixed(4)}</td>
      `;
      fusedRanksBody.appendChild(tr);
    });
  } else {
    fusedRanksBody.innerHTML = `<tr><td colspan="3" class="empty-row">No rank data (Hybrid mode disabled)</td></tr>`;
  }

  // Render Dense Ranks
  denseRanksBody.innerHTML = "";
  if (debug.dense_results && debug.dense_results.length > 0) {
    debug.dense_results.forEach((item, index) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${index + 1}</td>
        <td>${escapeHtml(item.chunk_id)}</td>
        <td>${item.score.toFixed(4)}</td>
      `;
      denseRanksBody.appendChild(tr);
    });
  } else {
    denseRanksBody.innerHTML = `<tr><td colspan="3" class="empty-row">No dense rankings</td></tr>`;
  }

  // Render Sparse Ranks
  sparseRanksBody.innerHTML = "";
  if (debug.sparse_results && debug.sparse_results.length > 0) {
    debug.sparse_results.forEach((item, index) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${index + 1}</td>
        <td>${escapeHtml(item.chunk_id)}</td>
        <td>${item.score.toFixed(4)}</td>
      `;
      sparseRanksBody.appendChild(tr);
    });
  } else {
    sparseRanksBody.innerHTML = `<tr><td colspan="3" class="empty-row">No sparse rankings</td></tr>`;
  }
}

// ------------------------------------------------------------------
// INITIALIZATION
// ------------------------------------------------------------------
async function checkActiveSession() {
  try {
    const progressResponse = await fetch("/session/progress");
    if (progressResponse.ok) {
      const progressData = await progressResponse.json();
      if (
        progressData.stage && 
        progressData.stage !== "idle" && 
        progressData.stage !== "ready" && 
        progressData.stage !== "error"
      ) {
        // Build is currently running, jump directly to Screen 2 progress
        topicSetupScreen.classList.add("is-hidden");
        kbBuilderScreen.classList.remove("is-hidden");
        startProgressPolling();
        return;
      }
    }

    const statusResponse = await fetch("/session/status");
    if (statusResponse.ok) {
      const statusData = await statusResponse.json();
      if (statusData.ready) {
        // Topic already parsed, jump directly to Screen 3 chat workspace
        topicSetupScreen.classList.add("is-hidden");
        chatWorkspaceScreen.classList.remove("is-hidden");
        await loadWorkspaceDetails();
      }
    }
  } catch (error) {
    console.error("Active session initialization check failed:", error);
  }
}

// Close PDF Viewer Panel and return to Stats panel
closePdfBtn.addEventListener("click", () => {
  pdfViewerIframe.src = "";
  pdfViewerSidebar.classList.add("is-hidden");
  workspaceInfoSidebar.classList.remove("is-hidden");
  
  // Reset active state on paper items
  document.querySelectorAll(".paper-item").forEach(el => el.classList.remove("active"));
  const paperItems = document.querySelectorAll(".paper-item");
  const lastItem = paperItems[paperItems.length - 1];
  if (lastItem) lastItem.classList.add("active");
  
  activePaperId = null;
  activePaperTitle = null;
  activeScopeBadge.className = "scope-badge global";
  activeScopeBadge.textContent = "Global Scope";
  questionInput.placeholder = "Ask a question about all papers...";
});

checkActiveSession();

// Start keep-alive heartbeat loop to prevent server shutdown
function startHeartbeat() {
  // Send heartbeat immediately on page load
  fetch("/heartbeat", { method: "POST" }).catch(() => {});
  
  // Periodic ping every 2 seconds
  setInterval(() => {
    fetch("/heartbeat", { method: "POST" }).catch(() => {});
  }, 2000);
}

startHeartbeat();
