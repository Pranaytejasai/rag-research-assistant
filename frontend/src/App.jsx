import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// generate or reuse a unique session ID for this browser tab
function getSessionId() {
  let id = sessionStorage.getItem('grounded_session')
  if (!id) {
    id = 'sess_' + Math.random().toString(36).slice(2) + Date.now().toString(36)
    sessionStorage.setItem('grounded_session', id)
  }
  return id
}
const SESSION_ID = getSessionId()
const H = { 'Content-Type': 'application/json', 'X-Session-Id': SESSION_ID }
const HG = { 'X-Session-Id': SESSION_ID }

const LANGUAGES = ['English', 'Spanish', 'French', 'German', 'Hindi', 'Telugu',
  'Chinese', 'Arabic', 'Portuguese', 'Japanese', 'Italian', 'Russian']

function toText(val) {
  if (val == null) return ''
  if (typeof val === 'string') return val
  if (val.answer && typeof val.answer === 'string') {
    let text = val.answer
    if (val.citations && Array.isArray(val.citations) && val.citations.length) {
      text += '\n\n**Citations:**\n' + val.citations.map(c => `- ${c}`).join('\n')
    }
    return text
  }
  if (typeof val === 'object') {
    return Object.entries(val)
      .map(([key, value]) => {
        if (typeof value === 'number') return `**${key}**: ${value}% similar`
        return `### ${key}\n\n${value}`
      })
      .join('\n\n---\n\n')
  }
  return String(val)
}

function Citation({ text, sources }) {
  const arxivMatch = text.match(/arXiv:(\d{4}\.\d{4,5})/i)
  if (arxivMatch) {
    return <a href={`https://arxiv.org/abs/${arxivMatch[1]}`} target="_blank" rel="noreferrer" className="cite-link">{text}</a>
  }
  const matchKey = Object.keys(sources || {}).find(stem => text.includes(stem))
  if (matchKey && sources[matchKey]) {
    return <a href={sources[matchKey]} target="_blank" rel="noreferrer" className="cite-link">{text}</a>
  }
  return <span>{text}</span>
}

function TrustBadge({ v }) {
  if (!v) return null
  const color = v.level === 'high' ? 'var(--cite)' : v.level === 'medium' ? '#F2C14E' : 'var(--flag)'
  return (
    <div className="trust-block">
      <div className="trust" style={{ borderColor: color }}>
        <span style={{ color }}>
          ● Trust {v.score}% — {v.claims_supported}/{v.claims_total} claims supported
        </span>
      </div>
      {v.assessment && <div className="assessment">{v.assessment}</div>}
    </div>
  )
}

function MindMapGraph({ connections }) {
  const nodes = {}
  connections.forEach(([a, b]) => { nodes[a] = true; nodes[b] = true })
  const nodeList = Object.keys(nodes)
  const W = 900, H = 560, cx = W / 2, cy = H / 2
  const radius = Math.min(W, H) / 2 - 90
  const pos = {}
  nodeList.forEach((n, i) => {
    const angle = (i / nodeList.length) * 2 * Math.PI - Math.PI / 2
    pos[n] = { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) }
  })
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mm-svg">
      {connections.map(([a, b], i) => {
        const p1 = pos[a], p2 = pos[b]
        if (!p1 || !p2) return null
        return <line key={i} x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y}
          stroke="var(--line-2)" strokeWidth="1.5" opacity="0.6" />
      })}
      {nodeList.map((n, i) => {
        const p = pos[n]
        return (
          <g key={i}>
            <circle cx={p.x} cy={p.y} r="6" fill="var(--cite)" />
            <text x={p.x} y={p.y - 14} textAnchor="middle"
              fill="var(--paper)" fontSize="13" fontFamily="Space Grotesk">{n}</text>
          </g>
        )
      })}
    </svg>
  )
}

function EmptyState({ message }) {
  return (
    <div className="empty-state">
      <div className="empty-glyph"></div>
      <p>{message}</p>
    </div>
  )
}

const TABS = [
  { id: 'ask', label: 'Ask' },
  { id: 'compare', label: 'Compare' },
  { id: 'summaries', label: 'Summaries' },
  { id: 'contradictions', label: 'Contradictions' },
  { id: 'findings', label: 'Key Findings' },
  { id: 'gaps', label: 'Research Gaps' },
  { id: 'hypotheses', label: 'Hypotheses' },
  { id: 'review', label: 'Literature Review' },
  { id: 'similarity', label: 'Similarity' },
  { id: 'mindmap', label: 'Mind Map' },
  { id: 'timeline', label: 'Timeline' },
  { id: 'arxiv', label: 'ArXiv' },
  { id: 'pubmed', label: 'PubMed' },
  { id: 'crossref', label: 'CrossRef' },
  { id: 'alerts', label: 'Alerts' },
]

const NEED_PAPERS = ['ask', 'compare', 'summaries', 'contradictions', 'findings',
  'gaps', 'hypotheses', 'review', 'similarity', 'mindmap', 'timeline']
const NEED_TWO = ['compare', 'similarity']

function App() {
  const [papers, setPapers] = useState([])
  const [paperSources, setPaperSources] = useState({})
  const [uploading, setUploading] = useState(false)
  const [tab, setTab] = useState('ask')

  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState(null)
  const [compareTopic, setCompareTopic] = useState('')
  const [loading, setLoading] = useState(false)
  const [output, setOutput] = useState('')
  const [outputVerify, setOutputVerify] = useState(null)
  const [simData, setSimData] = useState(null)
  const [mindmap, setMindmap] = useState(null)
  const [timeline, setTimeline] = useState(null)
  const [reviewLength, setReviewLength] = useState('medium')

  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [loadingId, setLoadingId] = useState(null)
  const [citeData, setCiteData] = useState(null)
  const [citingId, setCitingId] = useState(null)

  const [maxResults, setMaxResults] = useState(5)
  const [fromYear, setFromYear] = useState('')
  const [toYear, setToYear] = useState('')

  const [alertEmail, setAlertEmail] = useState('')
  const [emailStatus, setEmailStatus] = useState('')

  const [language, setLanguage] = useState('English')
  const [displayAnswer, setDisplayAnswer] = useState('')
  const [translating, setTranslating] = useState(false)
  const [speaking, setSpeaking] = useState(false)
  const [audioUrl, setAudioUrl] = useState(null)
  const [recording, setRecording] = useState(false)
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const answerRef = useRef(null)

  const refreshSources = async () => {
    try {
      const res = await fetch(`${API}/paper-sources`, { headers: HG })
      const data = await res.json()
      setPaperSources(data.sources || {})
    } catch {}
  }

  useEffect(() => {
    fetch(`${API}/papers`, { headers: HG }).then(r => r.json()).then(d => setPapers(d.papers || [])).catch(() => {})
    refreshSources()
  }, [])

  useEffect(() => {
    if ((answer || output || simData) && answerRef.current) {
      answerRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [answer, output, simData])

  useEffect(() => {
    const run = async () => {
      if (!answer?.answer) { setDisplayAnswer(''); return }
      if (language === 'English') { setDisplayAnswer(answer.answer); return }
      setTranslating(true)
      try {
        const res = await fetch(`${API}/translate`, {
          method: 'POST', headers: H,
          body: JSON.stringify({ text: answer.answer, language })
        })
        const data = await res.json()
        setDisplayAnswer(data.result || answer.answer)
      } catch { setDisplayAnswer(answer.answer) }
      setTranslating(false)
    }
    run()
  }, [answer, language])

  const handleUpload = async (e) => {
    const files = Array.from(e.target.files)
    if (files.length === 0) return
    setUploading(true)
    for (const file of files) {
      const fd = new FormData()
      fd.append('file', file)
      try {
        const res = await fetch(`${API}/upload`, { method: 'POST', body: fd, headers: HG })
        const data = await res.json()
        setPapers(data.papers || [])
      } catch (err) { alert('Upload failed: ' + file.name) }
    }
    refreshSources()
    setUploading(false)
    e.target.value = ''
  }

  const removePaper = async (filename) => {
    try {
      const res = await fetch(`${API}/remove-paper`, {
        method: 'POST', headers: H,
        body: JSON.stringify({ filename })
      })
      const data = await res.json()
      setPapers(data.papers || [])
      refreshSources()
    } catch { alert('Remove failed') }
  }

  const resetLibrary = async () => {
    if (!confirm('Remove all papers from your library?')) return
    try {
      const res = await fetch(`${API}/reset-library`, { method: 'POST', headers: H })
      const data = await res.json()
      setPapers(data.papers || [])
      setPaperSources({})
    } catch { alert('Reset failed') }
  }

  const handleAsk = async () => {
    if (!question.trim()) return
    setLoading(true); setAnswer(null); setAudioUrl(null)
    try {
      const res = await fetch(`${API}/ask`, {
        method: 'POST', headers: H,
        body: JSON.stringify({ question })
      })
      setAnswer(await res.json())
    } catch { setAnswer({ answer: 'Connection error. Make sure the backend is running.', citations: [] }) }
    setLoading(false)
  }

  const handleListen = async () => {
    if (!displayAnswer) return
    setSpeaking(true); setAudioUrl(null)
    try {
      const res = await fetch(`${API}/text-to-speech`, {
        method: 'POST', headers: H,
        body: JSON.stringify({ text: displayAnswer })
      })
      const data = await res.json()
      if (data.audio) {
        const blob = new Blob([Uint8Array.from(atob(data.audio), c => c.charCodeAt(0))], { type: 'audio/mp3' })
        setAudioUrl(URL.createObjectURL(blob))
      }
    } catch { alert('Voice generation failed') }
    setSpeaking(false)
  }

  const toggleRecording = async () => {
    if (recording) {
      mediaRecorderRef.current?.stop()
      setRecording(false)
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mr = new MediaRecorder(stream)
      chunksRef.current = []
      mr.ondataavailable = e => chunksRef.current.push(e.data)
      mr.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const fd = new FormData()
        fd.append('file', blob, 'recording.webm')
        setLoading(true)
        try {
          const res = await fetch(`${API}/transcribe`, { method: 'POST', body: fd, headers: HG })
          const data = await res.json()
          if (data.text) setQuestion(data.text)
        } catch { alert('Transcription failed') }
        setLoading(false)
        stream.getTracks().forEach(t => t.stop())
      }
      mediaRecorderRef.current = mr
      mr.start()
      setRecording(true)
    } catch { alert('Microphone access denied') }
  }

  const runFeature = async (endpoint) => {
    setLoading(true); setOutput(''); setOutputVerify(null)
    try {
      const res = await fetch(`${API}/${endpoint}`, { headers: HG })
      const data = await res.json()
      setOutput(data.result ?? 'No result.')
      if (data.verification) setOutputVerify(data.verification)
    } catch { setOutput('Connection error. Make sure the backend is running.') }
    setLoading(false)
  }

  const handleCompare = async () => {
    if (!compareTopic.trim()) return
    setLoading(true); setOutput(''); setOutputVerify(null)
    try {
      const res = await fetch(`${API}/compare`, {
        method: 'POST', headers: H,
        body: JSON.stringify({ topic: compareTopic })
      })
      const data = await res.json()
      setOutput(data.result ?? 'No result.')
      if (data.verification) setOutputVerify(data.verification)
    } catch { setOutput('Connection error.') }
    setLoading(false)
  }

  const runReview = async () => {
    setLoading(true); setOutput(''); setOutputVerify(null)
    try {
      const res = await fetch(`${API}/literature-review`, {
        method: 'POST', headers: H,
        body: JSON.stringify({ length: reviewLength })
      })
      const data = await res.json()
      setOutput(data.result ?? 'No result.')
      if (data.verification) setOutputVerify(data.verification)
    } catch { setOutput('Connection error.') }
    setLoading(false)
  }

  const runSimilarity = async () => {
    setLoading(true); setSimData(null)
    try {
      const res = await fetch(`${API}/similarity`, { headers: HG })
      const data = await res.json()
      setSimData(data.result || {})
    } catch { setSimData(null) }
    setLoading(false)
  }

  const runMindmap = async () => {
    setLoading(true); setMindmap(null)
    try {
      const res = await fetch(`${API}/mindmap`, { headers: HG })
      const data = await res.json()
      setMindmap(data.connections || [])
    } catch { setMindmap([]) }
    setLoading(false)
  }

  const runTimeline = async () => {
    setLoading(true); setTimeline(null)
    try {
      const res = await fetch(`${API}/timeline`, { headers: HG })
      const data = await res.json()
      setTimeline(data.timeline || [])
    } catch { setTimeline([]) }
    setLoading(false)
  }

  const doSearch = async (source) => {
    if (!searchQuery.trim()) return
    setLoading(true); setSearchResults([]); setCiteData(null)
    try {
      const body = {
        query: searchQuery,
        max_results: parseInt(maxResults) || 5,
        from_year: fromYear ? parseInt(fromYear) : null,
        to_year: toYear ? parseInt(toYear) : null
      }
      const res = await fetch(`${API}/search-${source}`, {
        method: 'POST', headers: H,
        body: JSON.stringify(body)
      })
      const data = await res.json()
      setSearchResults(data.results || [])
    } catch { setSearchResults([]) }
    setLoading(false)
  }

  const loadPaper = async (source, paper) => {
    setLoadingId(paper.arxiv_id || paper.pmid || paper.doi || paper.title)
    try {
      let body = source === 'arxiv' ? { pdf_url: paper.pdf_url, arxiv_id: paper.arxiv_id } : paper
      const res = await fetch(`${API}/load-${source}`, {
        method: 'POST', headers: H,
        body: JSON.stringify(body)
      })
      const data = await res.json()
      setPapers(data.papers || [])
      refreshSources()
    } catch (err) { alert('Load failed') }
    setLoadingId(null)
  }

  const getCite = async (paper) => {
    const pid = paper.arxiv_id || paper.pmid || paper.doi || paper.title
    setCitingId(pid); setCiteData(null)
    try {
      const url = paper.arxiv_id ? `https://arxiv.org/abs/${paper.arxiv_id}` :
        paper.pubmed_url || paper.doi_url || ''
      const res = await fetch(`${API}/cite`, {
        method: 'POST', headers: H,
        body: JSON.stringify({
          title: paper.title, authors: paper.authors,
          published: paper.published, journal: paper.journal || '',
          doi: paper.doi || '', url
        })
      })
      const data = await res.json()
      setCiteData({ id: pid, citations: data.citations || {} })
    } catch { alert('Citation failed') }
    setCitingId(null)
  }

  const copyText = (text) => { navigator.clipboard.writeText(text) }

  const checkAlerts = async () => {
    if (!searchQuery.trim()) return
    setLoading(true); setSearchResults([]); setEmailStatus('')
    try {
      await fetch(`${API}/alert-add`, {
        method: 'POST', headers: H,
        body: JSON.stringify({ topic: searchQuery })
      })
      const res = await fetch(`${API}/alert-check`, {
        method: 'POST', headers: H,
        body: JSON.stringify({ topic: searchQuery })
      })
      const data = await res.json()
      setSearchResults(data.new || [])
    } catch { setSearchResults([]) }
    setLoading(false)
  }

  const sendAlertEmail = async () => {
    if (!alertEmail.trim() || !searchQuery.trim()) {
      setEmailStatus('Enter a topic and email first')
      return
    }
    setEmailStatus('Sending… (checking all sources)')
    try {
      const res = await fetch(`${API}/alert-email`, {
        method: 'POST', headers: H,
        body: JSON.stringify({ email: alertEmail, topic: searchQuery })
      })
      const data = await res.json()
      if (data.success) {
        setEmailStatus(`✓ Sent ${data.count} papers to ${alertEmail}`)
      } else {
        setEmailStatus(`Failed: ${data.error || 'unknown error'}`)
      }
    } catch {
      setEmailStatus('Failed to send')
    }
  }

  const trustColor = (level) => {
    if (level === 'high') return 'var(--cite)'
    if (level === 'medium') return '#F2C14E'
    return 'var(--flag)'
  }

  const resetView = () => {
    setOutput(''); setOutputVerify(null); setSimData(null); setSearchResults([]); setMindmap(null); setTimeline(null); setAnswer(null); setCiteData(null); setEmailStatus('');
  }

  const paperCount = papers.length
  const blockedNoPapers = NEED_PAPERS.includes(tab) && paperCount === 0
  const blockedNeedTwo = NEED_TWO.includes(tab) && paperCount < 2

  return (
    <div className="app">
      <nav className="nav">
        <div className="brand"><span className="glyph"></span> Grounded</div>
        <div className="nav-meta">RAG Research Assistant</div>
      </nav>

      <div className="layout">
        <aside className="sidebar">
          <div className="side-label">Your Papers {paperCount > 0 && `· ${paperCount}`}</div>
          <label className="upload-btn">
            {uploading ? 'Uploading…' : '+ Upload paper'}
            <input type="file" accept=".pdf" multiple onChange={handleUpload} hidden />
          </label>
          <div className="paper-list">
            {papers.length === 0 && <div className="empty">No papers yet</div>}
            {papers.map((p, i) => (
              <div className="paper-item" key={i}>
                <span className="dot"></span>
                <span className="paper-name">{p}</span>
                <button className="paper-remove" onClick={() => removePaper(p)} title="Remove">✕</button>
              </div>
            ))}
            {papers.length > 0 && (
              <button className="reset-link" onClick={resetLibrary}>Clear all papers</button>
            )}
          </div>
        </aside>

        <main className="main">
          <div className="tabs">
            {TABS.map(t => (
              <button key={t.id}
                className={`tab ${tab === t.id ? 'active' : ''}`}
                onClick={() => { setTab(t.id); resetView(); }}>
                {t.label}
              </button>
            ))}
          </div>

          {blockedNoPapers && (
            <>
              <div className="hero-line"><h1>{TABS.find(t => t.id === tab)?.label}</h1></div>
              <EmptyState message="Upload a paper from the sidebar to get started." />
            </>
          )}
          {!blockedNoPapers && blockedNeedTwo && (
            <>
              <div className="hero-line"><h1>{TABS.find(t => t.id === tab)?.label}</h1></div>
              <EmptyState message="This feature needs at least 2 papers. Upload one more to compare." />
            </>
          )}

          {tab === 'ask' && !blockedNoPapers && (
            <>
              <div className="hero-line"><h1>Ask your <em>library</em>.</h1></div>
              <div className="ask-controls">
                <select className="lang-select" value={language} onChange={e => setLanguage(e.target.value)}>
                  {LANGUAGES.map(l => <option key={l} value={l}>🌍 {l}</option>)}
                </select>
                <button className={`mic-btn ${recording ? 'rec' : ''}`} onClick={toggleRecording}>
                  {recording ? '⏹ Stop' : '🎤 Speak'}
                </button>
              </div>
              <div className="ask-box">
                <input type="text" placeholder="Ask a question across your papers…"
                  value={question} onChange={e => setQuestion(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleAsk()} />
                <button onClick={handleAsk} disabled={loading}>
                  {loading ? 'Thinking…' : 'Ask →'}
                </button>
              </div>
              {answer && (
                <div className="answer-card" ref={answerRef}>
                  <div className="answer-label">Answer {translating && '· translating…'}</div>
                  <div className="answer-text"><ReactMarkdown>{toText(displayAnswer)}</ReactMarkdown></div>
                  <div className="voice-row">
                    <button className="listen-btn" onClick={handleListen} disabled={speaking}>
                      {speaking ? '🔊 Generating…' : '🔊 Listen'}
                    </button>
                    {audioUrl && <audio controls src={audioUrl} className="audio-player" />}
                  </div>
                  {answer.verification && (
                    <div className="trust-block" style={{ marginTop: 20 }}>
                      <div className="trust" style={{ borderColor: trustColor(answer.verification.level) }}>
                        <span style={{ color: trustColor(answer.verification.level) }}>
                          ● Trust {answer.verification.score}% — {answer.verification.claims_supported}/{answer.verification.claims_total} claims supported
                        </span>
                      </div>
                      {answer.verification.assessment && <div className="assessment">{answer.verification.assessment}</div>}
                    </div>
                  )}
                  {answer.citations?.length > 0 && (
                    <div className="citations">
                      <div className="cite-label">Citations</div>
                      {answer.citations.map((c, i) => <div className="cite" key={i}><Citation text={c} sources={paperSources} /></div>)}
                    </div>
                  )}
                </div>
              )}
            </>
          )}

          {tab === 'compare' && !blockedNoPapers && !blockedNeedTwo && (
            <>
              <div className="hero-line"><h1>Compare <em>papers</em>.</h1></div>
              <div className="ask-box">
                <input type="text" placeholder="Enter a topic to compare across papers…"
                  value={compareTopic} onChange={e => setCompareTopic(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleCompare()} />
                <button onClick={handleCompare} disabled={loading}>
                  {loading ? 'Comparing…' : 'Compare →'}
                </button>
              </div>
              {output && (
                <div className="answer-card" ref={answerRef}>
                  <TrustBadge v={outputVerify} />
                  <div className="answer-text"><ReactMarkdown>{toText(output)}</ReactMarkdown></div>
                </div>
              )}
            </>
          )}

          {['summaries', 'contradictions', 'findings', 'gaps', 'hypotheses'].includes(tab) && !blockedNoPapers && (
            <>
              <div className="hero-line"><h1>{TABS.find(t => t.id === tab)?.label}</h1></div>
              <button className="run-btn" onClick={() => runFeature(
                tab === 'findings' ? 'key-findings' : tab
              )} disabled={loading}>
                {loading ? 'Working…' : `Generate ${TABS.find(t => t.id === tab)?.label} →`}
              </button>
              {output && (
                <div className="answer-card" ref={answerRef}>
                  <TrustBadge v={outputVerify} />
                  <div className="answer-text"><ReactMarkdown>{toText(output)}</ReactMarkdown></div>
                </div>
              )}
            </>
          )}

          {tab === 'review' && !blockedNoPapers && (
            <>
              <div className="hero-line"><h1>Literature <em>Review</em>.</h1></div>
              <div className="ask-controls">
                <select className="lang-select" value={reviewLength} onChange={e => setReviewLength(e.target.value)}>
                  <option value="short">Short — 1 paragraph per section</option>
                  <option value="medium">Medium — balanced</option>
                  <option value="detailed">Detailed — in-depth</option>
                  <option value="comprehensive">Comprehensive — thesis-style</option>
                </select>
              </div>
              <button className="run-btn" onClick={runReview} disabled={loading}>
                {loading ? 'Writing…' : 'Generate Literature Review →'}
              </button>
              {output && (
                <div className="answer-card" ref={answerRef}>
                  <TrustBadge v={outputVerify} />
                  <div className="answer-text"><ReactMarkdown>{toText(output)}</ReactMarkdown></div>
                </div>
              )}
            </>
          )}

          {tab === 'similarity' && !blockedNoPapers && !blockedNeedTwo && (
            <>
              <div className="hero-line"><h1>Paper <em>Similarity</em>.</h1></div>
              <button className="run-btn" onClick={runSimilarity} disabled={loading}>
                {loading ? 'Calculating…' : 'Calculate Similarity →'}
              </button>
              {simData && Object.keys(simData).length > 0 && (
                <div className="answer-card" ref={answerRef}>
                  {Object.entries(simData).map(([pair, score], i) => (
                    <div className="sim-row" key={i}>
                      <div className="sim-labels">
                        <span>{pair}</span>
                        <span className="sim-score">{score}%</span>
                      </div>
                      <div className="sim-track">
                        <div className="sim-fill" style={{ width: `${score}%` }}></div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          {tab === 'mindmap' && !blockedNoPapers && (
            <>
              <div className="hero-line"><h1>Concept <em>Map</em>.</h1></div>
              <button className="run-btn" onClick={runMindmap} disabled={loading}>
                {loading ? 'Building…' : 'Build Mind Map →'}
              </button>
              {mindmap && mindmap.length > 0 && (
                <>
                  <div className="answer-card">
                    <MindMapGraph connections={mindmap} />
                  </div>
                  <div className="answer-card" style={{ marginTop: '20px' }}>
                    <div className="answer-label">Connections</div>
                    <div className="mindmap">
                      {mindmap.map(([from, to], i) => (
                        <div className="mm-edge" key={i}>
                          <span className="mm-node">{from}</span>
                          <span className="mm-arrow">→</span>
                          <span className="mm-node mm-node-2">{to}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </>
          )}

          {tab === 'timeline' && !blockedNoPapers && (
            <>
              <div className="hero-line"><h1>Research <em>Timeline</em>.</h1></div>
              <button className="run-btn" onClick={runTimeline} disabled={loading}>
                {loading ? 'Building…' : 'Build Timeline →'}
              </button>
              {timeline && timeline.length > 0 && (
                <div className="timeline">
                  {timeline.map((item, i) => (
                    <div className="tl-item" key={i}>
                      <div className="tl-marker"><span className="tl-year">{item.year}</span></div>
                      <div className="tl-content">
                        <div className="tl-paper">{item.paper}</div>
                        <div className="tl-contrib">{item.contribution}</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          {['arxiv', 'pubmed', 'crossref'].includes(tab) && (
            <>
              <div className="hero-line">
                <h1>Search <em>{tab === 'arxiv' ? 'ArXiv' : tab === 'pubmed' ? 'PubMed' : 'CrossRef'}</em>.</h1>
              </div>
              <div className="search-filters">
                <div className="filter-group">
                  <span className="filter-label">Papers</span>
                  <select className="filter-select" value={maxResults} onChange={e => setMaxResults(e.target.value)}>
                    <option value="5">5</option>
                    <option value="10">10</option>
                    <option value="15">15</option>
                    <option value="20">20</option>
                  </select>
                </div>
                <div className="filter-group">
                  <span className="filter-label">From</span>
                  <input className="filter-input" type="number" placeholder="2015"
                    value={fromYear} onChange={e => setFromYear(e.target.value)} />
                </div>
                <div className="filter-group">
                  <span className="filter-label">To</span>
                  <input className="filter-input" type="number" placeholder="2026"
                    value={toYear} onChange={e => setToYear(e.target.value)} />
                </div>
              </div>
              <div className="ask-box">
                <input type="text" placeholder={`Search ${tab} for papers…`}
                  value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && doSearch(tab)} />
                <button onClick={() => doSearch(tab)} disabled={loading}>
                  {loading ? 'Searching…' : 'Search →'}
                </button>
              </div>
              <div className="results">
                {searchResults.map((paper, i) => {
                  const pid = paper.arxiv_id || paper.pmid || paper.doi || paper.title
                  const link = paper.arxiv_id ? `https://arxiv.org/abs/${paper.arxiv_id}` :
                    paper.pubmed_url || paper.doi_url || null
                  return (
                    <div className="result-card" key={i}>
                      <div className="result-title">{paper.title}</div>
                      <div className="result-meta">
                        {paper.authors?.slice(0, 3).join(', ')} · {paper.published}
                      </div>
                      <div className="result-summary">{(paper.summary || '').slice(0, 220)}…</div>
                      <div className="result-actions">
                        <button className="load-btn" onClick={() => loadPaper(tab, paper)}
                          disabled={loadingId === pid}>
                          {loadingId === pid ? 'Loading…' : '+ Add to library'}
                        </button>
                        {link && <a className="view-btn" href={link} target="_blank" rel="noreferrer">View paper ↗</a>}
                        <button className="view-btn" onClick={() => getCite(paper)} disabled={citingId === pid}>
                          {citingId === pid ? 'Loading…' : '❞ Cite'}
                        </button>
                      </div>
                      {citeData && citeData.id === pid && (
                        <div className="cite-box">
                          {Object.entries(citeData.citations).map(([format, text]) => (
                            <div className="cite-format" key={format}>
                              <div className="cite-format-head">
                                <span className="cite-format-name">{format}</span>
                                <button className="cite-copy" onClick={() => copyText(text)}>Copy</button>
                              </div>
                              <div className="cite-format-text">{text}</div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </>
          )}

          {tab === 'alerts' && (
            <>
              <div className="hero-line"><h1>Research <em>Alerts</em>.</h1></div>
              <div className="ask-box">
                <input type="text" placeholder="Enter a topic to track…"
                  value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && checkAlerts()} />
                <button onClick={checkAlerts} disabled={loading}>
                  {loading ? 'Checking…' : 'Check →'}
                </button>
              </div>
              <div className="email-alert-row">
                <input type="email" className="email-input" placeholder="your@email.com"
                  value={alertEmail} onChange={e => setAlertEmail(e.target.value)} />
                <button className="email-btn" onClick={sendAlertEmail}>📧 Email me these papers</button>
              </div>
              {emailStatus && <div className="email-status">{emailStatus}</div>}
              <div className="results">
                {searchResults.map((paper, i) => {
                  const link = paper.arxiv_id ? `https://arxiv.org/abs/${paper.arxiv_id}` :
                    paper.pubmed_url || paper.doi_url || null
                  return (
                    <div className="result-card" key={i}>
                      <div className="result-title">🆕 {paper.title} {paper.source && <span className="src-tag">{paper.source}</span>}</div>
                      <div className="result-meta">
                        {paper.authors?.slice(0, 3).join(', ')} · {paper.published}
                      </div>
                      <div className="result-summary">{(paper.summary || '').slice(0, 220)}…</div>
                      {link && <div className="result-actions"><a className="view-btn" href={link} target="_blank" rel="noreferrer">View paper ↗</a></div>}
                    </div>
                  )
                })}
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  )
}

export default App