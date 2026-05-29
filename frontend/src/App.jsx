import { useState, useRef, useEffect } from "react";
import "./App.css";

const DEV = import.meta.env.DEV;
const API_BASE = DEV ? "http://localhost:8000" : "";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [activeDoc, setActiveDoc] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState(null);
  const fileInputRef = useRef(null);

  // On first load, ask the backend which document is currently loaded.
  useEffect(() => {
    fetch(`${API_BASE}/status`)
      .then((r) => r.json())
      .then((d) => setActiveDoc(d.document))
      .catch(() => {});
  }, []);

  async function askQuestion() {
    const trimmed = question.trim();
    if (!trimmed) return;

    setLoading(true);
    setError(null);
    setAnswer(null);
    setSources([]);

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });
      if (!response.ok) throw new Error(`Server responded with ${response.status}`);
      const data = await response.json();
      setAnswer(data.answer);
      setSources(data.sources);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleFileChange(e) {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setUploadMsg(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Upload failed (${response.status})`);

      setActiveDoc(data.document);
      setUploadMsg(`Indexed “${data.document}” — ${data.chunks} chunks ready.`);
      // a fresh document means old answers no longer apply
      setAnswer(null);
      setSources([]);
    } catch (err) {
      setUploadMsg(`⚠ ${err.message}`);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = ""; // allow re-uploading same file
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  }

  return (
    <div className="page">
      <header className="masthead">
        <p className="eyebrow">Retrieval-augmented document Q&amp;A</p>
        <h1>Ask the document.</h1>
        <p className="lede">
          Answers come only from the loaded document — and every answer shows
          the passages it was drawn from. Upload your own PDF to query it.
        </p>
      </header>

      <div className="docbar">
        <div className="docbar-info">
          <span className="docbar-label">Loaded</span>
          <span className="docbar-name">{activeDoc || "—"}</span>
        </div>
        <button
          className="docbar-upload"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? "Indexing…" : "Upload PDF"}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf"
          onChange={handleFileChange}
          style={{ display: "none" }}
        />
      </div>

      {uploadMsg && <div className="upload-msg">{uploadMsg}</div>}

      <div className="ask">
        <textarea
          className="ask-input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="e.g. How many vacation days do employees get?"
          rows={2}
        />
        <button className="ask-button" onClick={askQuestion} disabled={loading || uploading}>
          {loading ? "Thinking…" : "Ask"}
        </button>
      </div>

      {error && (
        <div className="error">
          Couldn’t reach the API: {error}. Is the backend running?
        </div>
      )}

      {loading && (
        <div className="status">
          <span className="dot" /> Searching the document and composing an answer…
        </div>
      )}

      {answer && (
        <section className="answer">
          <h2 className="section-label">Answer</h2>
          <p className="answer-text">{answer}</p>
        </section>
      )}

      {sources.length > 0 && (
        <section className="sources">
          <h2 className="section-label">Sources · {sources.length}</h2>
          <div className="source-list">
            {sources.map((s) => (
              <article className="source-card" key={s.rank}>
                <span className="source-rank">{s.rank}</span>
                <p className="source-text">{s.text}</p>
              </article>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

export default App;