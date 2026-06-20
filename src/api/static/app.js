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
      const preview = source.text.length > 260 ? `${source.text.slice(0, 260)}...` : source.text;
      item.innerHTML = `<strong>${index + 1}. ${source.paper_id || source.chunk_id}</strong> ` +
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

  statusLabel.textContent = "Preparing topic";
  topicHelp.textContent = `Searching arXiv, downloading PDFs, extracting text, and building the vector index. This can take several minutes for ${maxPapers} papers.`;
  topicForm.querySelector("button").disabled = true;

  try {
    const response = await fetch("/session/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        topic,
        max_papers: maxPapers,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Could not prepare topic");
    }

    topicSetup.classList.add("is-hidden");
    form.classList.remove("is-hidden");
    messages.innerHTML = "";
    addMessage(
      "assistant",
      `Ready for "${payload.topic}". Indexed ${payload.chunks_indexed} chunks from ${payload.pdfs_downloaded} downloaded PDFs.`
    );
    statusLabel.textContent = "Ready";
    questionInput.focus();
  } catch (error) {
    addMessage("assistant", error.message);
    statusLabel.textContent = "Needs attention";
    topicForm.querySelector("button").disabled = false;
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";
  statusLabel.textContent = "Searching";
  form.querySelector("button").disabled = true;

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: Number(topKInput.value || 5),
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed");
    }

    addMessage("assistant", payload.answer, payload.sources);
    statusLabel.textContent = "Ready";
  } catch (error) {
    addMessage("assistant", error.message);
    statusLabel.textContent = "Needs attention";
  } finally {
    form.querySelector("button").disabled = false;
    questionInput.focus();
  }
});
