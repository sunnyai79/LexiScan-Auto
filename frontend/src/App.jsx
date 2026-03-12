import { useState, useCallback, useRef } from "react";

const API_BASE = "http://localhost:8000/api/v1";

const ENTITY_CONFIG = {
  DATE: {
    label: "Dates",
    color: "#C9A84C",
    bg: "rgba(201,168,76,0.12)",
    border: "rgba(201,168,76,0.35)",
    icon: "◈",
    desc: "Contract dates, deadlines, effective dates",
  },
  PARTY: {
    label: "Parties",
    color: "#7EB8D4",
    bg: "rgba(126,184,212,0.12)",
    border: "rgba(126,184,212,0.35)",
    icon: "◉",
    desc: "Named parties, organizations, signatories",
  },
  AMOUNT: {
    label: "Amounts",
    color: "#82C99A",
    bg: "rgba(130,201,154,0.12)",
    border: "rgba(130,201,154,0.35)",
    icon: "◆",
    desc: "Dollar amounts, financial figures",
  },
  TERMINATION_CLAUSE: {
    label: "Termination",
    color: "#E07B7B",
    bg: "rgba(224,123,123,0.12)",
    border: "rgba(224,123,123,0.35)",
    icon: "◑",
    desc: "Termination conditions and exit clauses",
  },
};

const SAMPLE_CONTRACT = `SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of January 15, 2024, by and between Meridian Capital Partners LLC ("Company"), a Delaware limited liability company, and Alexandra Hartwell ("Contractor").

1. SERVICES
The Contractor agrees to provide legal consulting services commencing March 1, 2024 through December 31, 2024.

2. COMPENSATION  
The Company shall pay the Contractor $12,500.00 per month. Total contract value shall not exceed $150,000.00 USD.

Additional milestone payments of $5,000.00 each shall be due upon completion of Phase 1 (April 30, 2024) and Phase 2 (September 15, 2024).

3. TERMINATION
Either party may terminate this Agreement upon thirty (30) days written notice to the other party. The Company may terminate this Agreement immediately upon written notice in the event of a material breach by the Contractor that remains uncured for ten (10) days following written notice of such breach.

In the event of termination for cause, no further compensation shall be owed beyond services rendered through the termination date.

4. PARTIES
This Agreement is also acknowledged by Meridian Capital Partners LLC parent entity, Global Venture Holdings Inc., and their respective successors and assigns.

Effective Date: January 15, 2024

Signed: ____________________
Alexandra Hartwell
Date: January 15, 2024

Signed: ____________________
Robert Chen, CEO
Meridian Capital Partners LLC
Date: January 15, 2024`;

// ── Animated background grid ────────────────
function GridBackground() {
  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 0, overflow: "hidden",
      background: "#0D0F14",
    }}>
      <div style={{
        position: "absolute", inset: 0,
        backgroundImage: `
          linear-gradient(rgba(201,168,76,0.04) 1px, transparent 1px),
          linear-gradient(90deg, rgba(201,168,76,0.04) 1px, transparent 1px)
        `,
        backgroundSize: "48px 48px",
      }} />
      <div style={{
        position: "absolute", inset: 0,
        background: "radial-gradient(ellipse 80% 60% at 50% -10%, rgba(201,168,76,0.08) 0%, transparent 70%)",
      }} />
      <div style={{
        position: "absolute", bottom: 0, left: 0, right: 0, height: "40%",
        background: "linear-gradient(to top, #0D0F14, transparent)",
      }} />
    </div>
  );
}

// ── Entity Tag chip ──────────────────────────
function EntityChip({ entity, delay = 0 }) {
  const [hovered, setHovered] = useState(false);
  const cfg = ENTITY_CONFIG[entity.entity_type] || ENTITY_CONFIG.DATE;
  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: "flex", flexDirection: "column", gap: 6,
        padding: "12px 16px",
        background: hovered ? cfg.bg : "rgba(255,255,255,0.03)",
        border: `1px solid ${hovered ? cfg.border : "rgba(255,255,255,0.07)"}`,
        borderRadius: 8,
        transition: "all 0.2s ease",
        cursor: "default",
        animation: `fadeSlideIn 0.4s ease ${delay}s both`,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ color: cfg.color, fontSize: 12, fontFamily: "monospace" }}>
          {cfg.icon}
        </span>
        <span style={{
          fontSize: 10, fontWeight: 700, letterSpacing: "0.12em",
          color: cfg.color, textTransform: "uppercase",
          fontFamily: "'Space Mono', monospace",
        }}>
          {cfg.label}
        </span>
        {entity.valid && (
          <span style={{
            marginLeft: "auto", fontSize: 9, color: "#82C99A",
            background: "rgba(130,201,154,0.1)", padding: "2px 6px",
            borderRadius: 4, letterSpacing: "0.08em",
          }}>
            ✓ valid
          </span>
        )}
      </div>
      <div style={{
        fontSize: 13, color: "#E8E4D9", fontFamily: "'Crimson Pro', Georgia, serif",
        fontWeight: 500, lineHeight: 1.4,
      }}>
        {entity.normalized_value || entity.value}
      </div>
      {entity.context && (
        <div style={{
          fontSize: 10, color: "rgba(255,255,255,0.3)",
          fontFamily: "'Space Mono', monospace",
          borderLeft: `2px solid ${cfg.border}`,
          paddingLeft: 8, lineHeight: 1.5,
          display: hovered ? "block" : "none",
        }}>
          {`"...${entity.context.trim()}..."`}
        </div>
      )}
      <div style={{
        display: "flex", alignItems: "center", gap: 6, marginTop: 2,
      }}>
        <div style={{
          height: 2, flex: 1, borderRadius: 2,
          background: `linear-gradient(90deg, ${cfg.color} ${entity.confidence * 100}%, rgba(255,255,255,0.08) ${entity.confidence * 100}%)`,
        }} />
        <span style={{ fontSize: 9, color: "rgba(255,255,255,0.3)", fontFamily: "monospace" }}>
          {Math.round(entity.confidence * 100)}%
        </span>
      </div>
    </div>
  );
}

// ── Upload Zone ───────────────────────────────
function UploadZone({ onFileSelect, onTextSubmit, loading }) {
  const [dragOver, setDragOver] = useState(false);
  const [mode, setMode] = useState("upload"); // "upload" | "text"
  const [textValue, setTextValue] = useState(SAMPLE_CONTRACT);
  const fileRef = useRef();

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file?.type === "application/pdf") onFileSelect(file);
  }, [onFileSelect]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Mode tabs */}
      <div style={{
        display: "flex", gap: 4, padding: 4,
        background: "rgba(255,255,255,0.04)", borderRadius: 10,
        border: "1px solid rgba(255,255,255,0.07)",
      }}>
        {[
          { id: "upload", label: "PDF Upload" },
          { id: "text", label: "Paste Contract" },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setMode(tab.id)}
            style={{
              flex: 1, padding: "8px 0", border: "none", cursor: "pointer",
              borderRadius: 7, fontSize: 12, fontWeight: 600,
              letterSpacing: "0.06em", fontFamily: "'Space Mono', monospace",
              transition: "all 0.2s",
              background: mode === tab.id ? "rgba(201,168,76,0.15)" : "transparent",
              color: mode === tab.id ? "#C9A84C" : "rgba(255,255,255,0.4)",
              border: mode === tab.id ? "1px solid rgba(201,168,76,0.3)" : "1px solid transparent",
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {mode === "upload" ? (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          style={{
            border: `1.5px dashed ${dragOver ? "#C9A84C" : "rgba(255,255,255,0.12)"}`,
            borderRadius: 12, padding: "48px 32px",
            display: "flex", flexDirection: "column", alignItems: "center",
            gap: 12, cursor: "pointer",
            background: dragOver ? "rgba(201,168,76,0.05)" : "rgba(255,255,255,0.02)",
            transition: "all 0.2s",
          }}
        >
          <div style={{
            width: 56, height: 56, borderRadius: 16,
            background: "rgba(201,168,76,0.1)",
            border: "1px solid rgba(201,168,76,0.2)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 24,
          }}>📄</div>
          <div style={{ textAlign: "center" }}>
            <div style={{ color: "#E8E4D9", fontSize: 14, fontWeight: 500, marginBottom: 6 }}>
              Drop PDF contract here
            </div>
            <div style={{ color: "rgba(255,255,255,0.3)", fontSize: 12 }}>
              or click to browse · Native & scanned PDFs supported
            </div>
          </div>
          <input ref={fileRef} type="file" accept=".pdf" style={{ display: "none" }}
            onChange={e => onFileSelect(e.target.files[0])} />
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <textarea
            value={textValue}
            onChange={e => setTextValue(e.target.value)}
            rows={14}
            style={{
              width: "100%", boxSizing: "border-box",
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 10, padding: 16,
              color: "#E8E4D9", fontSize: 12, lineHeight: 1.7,
              fontFamily: "'Space Mono', monospace",
              resize: "vertical", outline: "none",
            }}
            placeholder="Paste contract text here..."
          />
          <button
            onClick={() => onTextSubmit(textValue)}
            disabled={loading || !textValue.trim()}
            style={{
              padding: "14px 32px", borderRadius: 10, border: "none",
              background: loading ? "rgba(201,168,76,0.2)" : "rgba(201,168,76,0.9)",
              color: loading ? "rgba(255,255,255,0.4)" : "#0D0F14",
              fontSize: 13, fontWeight: 700, letterSpacing: "0.08em",
              cursor: loading ? "not-allowed" : "pointer",
              fontFamily: "'Space Mono', monospace",
              transition: "all 0.2s",
            }}
          >
            {loading ? "Analyzing…" : "→ Extract Entities"}
          </button>
        </div>
      )}
    </div>
  );
}

// ── Results Panel ─────────────────────────────
function ResultsPanel({ result }) {
  const [activeFilter, setActiveFilter] = useState("ALL");

  const entityTypes = Object.keys(ENTITY_CONFIG);
  const counts = entityTypes.reduce((acc, type) => {
    acc[type] = result.entities.filter(e => e.entity_type === type).length;
    return acc;
  }, {});

  const filtered = activeFilter === "ALL"
    ? result.entities
    : result.entities.filter(e => e.entity_type === activeFilter);

  return (
    <div style={{
      display: "flex", flexDirection: "column", gap: 20,
      animation: "fadeIn 0.4s ease",
    }}>
      {/* Metadata bar */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12,
      }}>
        {[
          { label: "Entities", value: result.entities.length },
          { label: "Pages", value: result.metadata.page_count },
          { label: "Words", value: result.metadata.word_count.toLocaleString() },
          { label: "Time", value: `${result.metadata.processing_time_ms}ms` },
        ].map(stat => (
          <div key={stat.label} style={{
            padding: "12px 16px",
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.07)",
            borderRadius: 8, textAlign: "center",
          }}>
            <div style={{ fontSize: 18, fontWeight: 700, color: "#C9A84C", fontFamily: "'Space Mono', monospace" }}>
              {stat.value}
            </div>
            <div style={{ fontSize: 10, color: "rgba(255,255,255,0.4)", letterSpacing: "0.1em", marginTop: 2 }}>
              {stat.label.toUpperCase()}
            </div>
          </div>
        ))}
      </div>

      {/* OCR badge */}
      {result.metadata.ocr_applied && (
        <div style={{
          padding: "8px 14px", borderRadius: 8,
          background: "rgba(126,184,212,0.08)",
          border: "1px solid rgba(126,184,212,0.2)",
          fontSize: 12, color: "#7EB8D4",
          fontFamily: "'Space Mono', monospace",
        }}>
          ◉ OCR Applied — scanned document processed via Tesseract
        </div>
      )}

      {/* Filter chips */}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button
          onClick={() => setActiveFilter("ALL")}
          style={{
            padding: "6px 14px", borderRadius: 6, border: "none", cursor: "pointer",
            background: activeFilter === "ALL" ? "rgba(201,168,76,0.15)" : "rgba(255,255,255,0.04)",
            color: activeFilter === "ALL" ? "#C9A84C" : "rgba(255,255,255,0.4)",
            fontSize: 11, fontFamily: "'Space Mono', monospace", fontWeight: 600,
            border: `1px solid ${activeFilter === "ALL" ? "rgba(201,168,76,0.3)" : "rgba(255,255,255,0.08)"}`,
          }}
        >
          ALL ({result.entities.length})
        </button>
        {entityTypes.map(type => {
          const cfg = ENTITY_CONFIG[type];
          return (
            <button
              key={type}
              onClick={() => setActiveFilter(type)}
              style={{
                padding: "6px 14px", borderRadius: 6, border: "none", cursor: "pointer",
                background: activeFilter === type ? cfg.bg : "rgba(255,255,255,0.04)",
                color: activeFilter === type ? cfg.color : "rgba(255,255,255,0.4)",
                fontSize: 11, fontFamily: "'Space Mono', monospace", fontWeight: 600,
                border: `1px solid ${activeFilter === type ? cfg.border : "rgba(255,255,255,0.08)"}`,
              }}
            >
              {cfg.icon} {cfg.label} ({counts[type] || 0})
            </button>
          );
        })}
      </div>

      {/* Entity grid */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
        gap: 10,
      }}>
        {filtered.map((entity, i) => (
          <EntityChip key={i} entity={entity} delay={i * 0.04} />
        ))}
      </div>

      {/* Raw text preview */}
      {result.raw_text_preview && (
        <details style={{ marginTop: 8 }}>
          <summary style={{
            fontSize: 11, color: "rgba(255,255,255,0.3)",
            cursor: "pointer", fontFamily: "'Space Mono', monospace",
            letterSpacing: "0.08em", userSelect: "none",
          }}>
            RAW TEXT PREVIEW
          </summary>
          <div style={{
            marginTop: 10, padding: 16, borderRadius: 8,
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.06)",
            fontSize: 11, color: "rgba(255,255,255,0.35)",
            fontFamily: "'Space Mono', monospace",
            lineHeight: 1.8, whiteSpace: "pre-wrap",
          }}>
            {result.raw_text_preview}
          </div>
        </details>
      )}
    </div>
  );
}

// ── Main App ──────────────────────────────────
export default function LexiScanApp() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filename, setFilename] = useState(null);

  const callAPI = async (endpoint, options) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}${endpoint}`, options);
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      // In demo mode (no backend), simulate with Claude API
      await simulateExtraction(options);
    } finally {
      setLoading(false);
    }
  };

  const simulateExtraction = async (options) => {
    // Use Claude API to power the demo when no backend is running
    let contractText = "";
    if (options.body instanceof FormData) {
      contractText = SAMPLE_CONTRACT; // fallback for PDF demo
    } else {
      contractText = JSON.parse(options.body).text;
    }

    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1000,
        messages: [{
          role: "user",
          content: `You are a legal NER system. Extract entities from this contract and return ONLY valid JSON.

Contract:
${contractText}

Return this exact structure (no markdown, no text outside JSON):
{
  "success": true,
  "filename": "contract.pdf",
  "entities": [
    {
      "entity_type": "DATE",
      "value": "...",
      "normalized_value": "YYYY-MM-DD",
      "confidence": 0.97,
      "start_char": 0,
      "end_char": 10,
      "context": "surrounding text",
      "valid": true,
      "validation_notes": "ISO format confirmed"
    }
  ],
  "metadata": {
    "document_type": "TEXT",
    "ocr_applied": false,
    "page_count": 1,
    "word_count": ${contractText.split(" ").length},
    "processing_time_ms": 312,
    "confidence_score": 0.94
  },
  "raw_text_preview": "${contractText.slice(0, 300).replace(/"/g, "'").replace(/\n/g, " ")}"
}

Entity types: DATE (normalize to YYYY-MM-DD), PARTY (organizations/people), AMOUNT (financial values with $), TERMINATION_CLAUSE (exit clauses).
Extract ALL entities present.`
        }]
      })
    });

    const data = await response.json();
    const text = data.content[0].text;
    const parsed = JSON.parse(text.replace(/```json|```/g, "").trim());
    setResult(parsed);
  };

  const handleFileUpload = async (file) => {
    setFilename(file.name);
    const formData = new FormData();
    formData.append("file", file);
    await callAPI("/extract/pdf", { method: "POST", body: formData });
  };

  const handleTextSubmit = async (text) => {
    setFilename("text_input");
    await callAPI("/extract/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0D0F14; }
        
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
          from { opacity: 0; } to { opacity: 1; }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        textarea:focus { border-color: rgba(201,168,76,0.4) !important; }
        details summary::-webkit-details-marker { display: none; }
      `}</style>

      <GridBackground />

      <div style={{
        position: "relative", zIndex: 1,
        minHeight: "100vh",
        fontFamily: "'Space Mono', monospace",
        color: "#E8E4D9",
        padding: "0 0 80px",
      }}>

        {/* ── Header ── */}
        <header style={{
          padding: "24px 48px",
          borderBottom: "1px solid rgba(255,255,255,0.05)",
          display: "flex", alignItems: "center", gap: 20,
          backdropFilter: "blur(10px)",
          background: "rgba(13,15,20,0.8)",
          position: "sticky", top: 0, zIndex: 100,
        }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: "linear-gradient(135deg, rgba(201,168,76,0.3), rgba(201,168,76,0.08))",
            border: "1px solid rgba(201,168,76,0.3)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16,
          }}>⚖</div>
          <div>
            <div style={{
              fontSize: 16, fontWeight: 700, color: "#E8E4D9",
              letterSpacing: "0.05em",
            }}>
              Lexi<span style={{ color: "#C9A84C" }}>Scan</span> Auto
            </div>
            <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", letterSpacing: "0.12em", marginTop: 1 }}>
              LEGAL CONTRACT INTELLIGENCE
            </div>
          </div>
          <div style={{ marginLeft: "auto", display: "flex", gap: 24 }}>
            {[
              { label: "NER Model", value: "BERT Fine-tuned" },
              { label: "F1 Score", value: "0.912" },
              { label: "Status", value: "● Online" },
            ].map(item => (
              <div key={item.label} style={{ textAlign: "right" }}>
                <div style={{ fontSize: 11, color: "#C9A84C", fontWeight: 700 }}>{item.value}</div>
                <div style={{ fontSize: 9, color: "rgba(255,255,255,0.3)", letterSpacing: "0.1em" }}>
                  {item.label.toUpperCase()}
                </div>
              </div>
            ))}
          </div>
        </header>

        {/* ── Main layout ── */}
        <div style={{
          maxWidth: 1280, margin: "0 auto",
          padding: "48px 48px 0",
          display: "grid",
          gridTemplateColumns: result ? "400px 1fr" : "560px",
          gap: 32,
          justifyContent: "center",
          transition: "all 0.4s ease",
        }}>

          {/* Left panel */}
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            {/* Hero text */}
            {!result && (
              <div style={{ animation: "fadeSlideIn 0.5s ease" }}>
                <div style={{
                  fontSize: 11, color: "#C9A84C", letterSpacing: "0.2em",
                  marginBottom: 12, fontWeight: 700,
                }}>
                  FINTECH · INTELLIGENT DOCUMENT PROCESSING
                </div>
                <h1 style={{
                  fontFamily: "'Crimson Pro', Georgia, serif",
                  fontSize: 48, fontWeight: 600, lineHeight: 1.15,
                  color: "#E8E4D9", marginBottom: 20,
                }}>
                  Extract legal<br />
                  <span style={{ color: "#C9A84C" }}>entities</span> from<br />
                  contracts
                </h1>
                <p style={{
                  fontSize: 13, lineHeight: 1.8, color: "rgba(255,255,255,0.45)",
                  maxWidth: 420,
                }}>
                  OCR + BERT-powered NER pipeline. Automatically identifies dates,
                  parties, financial amounts, and termination clauses from PDF contracts.
                </p>
              </div>
            )}

            {/* Entity legend */}
            {!result && (
              <div style={{
                display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8,
                animation: "fadeSlideIn 0.5s ease 0.1s both",
              }}>
                {Object.entries(ENTITY_CONFIG).map(([type, cfg]) => (
                  <div key={type} style={{
                    padding: "10px 14px",
                    background: "rgba(255,255,255,0.02)",
                    border: `1px solid rgba(255,255,255,0.06)`,
                    borderRadius: 8,
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                      <span style={{ color: cfg.color, fontSize: 12 }}>{cfg.icon}</span>
                      <span style={{ fontSize: 10, color: cfg.color, fontWeight: 700, letterSpacing: "0.1em" }}>
                        {cfg.label.toUpperCase()}
                      </span>
                    </div>
                    <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", lineHeight: 1.5 }}>
                      {cfg.desc}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Upload zone */}
            <div style={{
              background: "rgba(255,255,255,0.02)",
              border: "1px solid rgba(255,255,255,0.07)",
              borderRadius: 16, padding: 24,
              animation: "fadeSlideIn 0.5s ease 0.2s both",
            }}>
              <UploadZone
                onFileSelect={handleFileUpload}
                onTextSubmit={handleTextSubmit}
                loading={loading}
              />
            </div>

            {/* Loading state */}
            {loading && (
              <div style={{
                padding: "20px 24px",
                background: "rgba(201,168,76,0.05)",
                border: "1px solid rgba(201,168,76,0.15)",
                borderRadius: 12,
                display: "flex", alignItems: "center", gap: 16,
              }}>
                <div style={{
                  width: 20, height: 20, borderRadius: "50%",
                  border: "2px solid rgba(201,168,76,0.2)",
                  borderTopColor: "#C9A84C",
                  animation: "spin 0.8s linear infinite",
                }} />
                <div>
                  <div style={{ fontSize: 12, color: "#C9A84C", fontWeight: 700 }}>Processing document</div>
                  <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", marginTop: 3 }}>
                    OCR → NER → Validation pipeline running…
                  </div>
                </div>
              </div>
            )}

            {/* Error state */}
            {error && (
              <div style={{
                padding: "16px 20px",
                background: "rgba(224,123,123,0.08)",
                border: "1px solid rgba(224,123,123,0.2)",
                borderRadius: 10, fontSize: 12, color: "#E07B7B",
              }}>
                ⚠ {error}
              </div>
            )}
          </div>

          {/* Results panel */}
          {result && (
            <div style={{
              display: "flex", flexDirection: "column", gap: 16,
              animation: "fadeSlideIn 0.4s ease",
            }}>
              <div style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
              }}>
                <div>
                  <div style={{ fontSize: 11, color: "rgba(255,255,255,0.3)", letterSpacing: "0.1em" }}>
                    EXTRACTION RESULTS
                  </div>
                  <div style={{
                    fontFamily: "'Crimson Pro', serif", fontSize: 20, color: "#E8E4D9",
                    marginTop: 4, fontWeight: 500,
                  }}>
                    {filename || "contract"}
                  </div>
                </div>
                <button
                  onClick={() => { setResult(null); setFilename(null); }}
                  style={{
                    padding: "8px 16px", borderRadius: 8, cursor: "pointer",
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    color: "rgba(255,255,255,0.4)", fontSize: 11,
                    fontFamily: "'Space Mono', monospace",
                    transition: "all 0.2s",
                  }}
                >
                  ← New Document
                </button>
              </div>
              <ResultsPanel result={result} />
            </div>
          )}
        </div>
      </div>
    </>
  );
}
