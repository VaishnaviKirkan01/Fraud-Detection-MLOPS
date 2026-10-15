import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity, AlertTriangle, ArrowRight, BarChart3, CheckCircle2, ChevronDown,
  Database, Download, FileSpreadsheet, Gauge, LayoutDashboard, Menu, Moon,
  RefreshCw, Search, Shield, ShieldAlert, ShieldCheck, Sun, TrendingUp, X,
  Zap,
} from "lucide-react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Line, LineChart,
  Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { checkApiHealth, getAnalyticsOverview, predictFraud } from "./api";
import "./App.css";

const NAV = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "predict", label: "Analyze transaction", icon: ShieldCheck },
  { id: "dataset", label: "PaySim explorer", icon: Database },
  { id: "analytics", label: "Dataset analytics", icon: TrendingUp },
];
const TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"];
const INITIAL_FORM = {
  step: 1, type: "TRANSFER", amount: 1000, oldbalanceOrg: 5000,
  newbalanceOrig: 4000, oldbalanceDest: 1000, newbalanceDest: 2000,
};
const money = (v) => new Intl.NumberFormat("en-IN", {
  style: "currency", currency: "INR", maximumFractionDigits: 0,
}).format(Number(v || 0));
const count = (v) => Number(v || 0).toLocaleString("en-IN");
const pct = (v, digits = 2) => `${Number(v || 0).toFixed(digits)}%`;
const outcomeIsFraud = (v) => String(v || "").toLowerCase() === "fraud" || Number(v) === 1;

function Stat({ icon: Icon, label, value, detail, tone = "blue" }) {
  return <article className={`stat-card tone-${tone}`}><div className="stat-top"><span>{label}</span><span className="stat-icon"><Icon size={18} /></span></div><strong className="stat-value">{value}</strong><div className="stat-detail">{detail}</div></article>;
}
function SectionTitle({ kicker, title, detail, action }) {
  return <div className="section-title"><div>{kicker && <div className="kicker">{kicker}</div>}<h2>{title}</h2>{detail && <p>{detail}</p>}</div>{action}</div>;
}
function Empty({ title, detail }) { return <div className="empty-state"><Database size={24}/><strong>{title}</strong><p>{detail}</p></div>; }
function Field({ label, name, value, onChange, min = "0", step = "any" }) {
  return <label className="field"><span>{label}</span><input type="number" min={min} step={step} value={value} onChange={(e) => onChange(name, e.target.value)} required /></label>;
}

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("fraudguard-theme") || "light");
  const [page, setPage] = useState("overview");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [health, setHealth] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [apiError, setApiError] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [form, setForm] = useState(INITIAL_FORM);
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [formError, setFormError] = useState("");
  const [query, setQuery] = useState("");
  const [labelFilter, setLabelFilter] = useState("ALL");

  useEffect(() => {
    localStorage.setItem("fraudguard-theme", theme);
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const [healthResult, analyticsResult] = await Promise.allSettled([
        checkApiHealth(), getAnalyticsOverview(),
      ]);
      if (healthResult.status === "fulfilled") setHealth(healthResult.value);
      if (analyticsResult.status === "fulfilled") setAnalytics(analyticsResult.value);
      const errors = [];
      if (healthResult.status === "rejected") errors.push("Health endpoint is unavailable.");
      if (analyticsResult.status === "rejected") errors.push("Dataset analytics could not be loaded.");
      setApiError(errors.join(" "));
    } finally { setRefreshing(false); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const sampleRows = analytics?.sample_transactions || [];
  const typeRows = useMemo(() => Object.entries(analytics?.transaction_types || {}).map(([name, value]) => ({ name, value: Number(value) })), [analytics]);
  const stepRows = useMemo(() => (analytics?.fraud_by_step || []).map((r) => ({
    ...r, label: `Step ${r.step}`, transactions: Number(r.transactions || 0),
    fraud_transactions: Number(r.fraud_transactions || 0), amount: Number(r.amount || 0),
  })), [analytics]);
  const filteredRows = useMemo(() => sampleRows.filter((row) => {
    const matchesText = `${row.dataset_id || ""} ${row.step ?? ""} ${row.type || ""} ${row.dataset_label || ""} ${row.amount ?? ""}`.toLowerCase().includes(query.toLowerCase());
    return matchesText && (labelFilter === "ALL" || String(row.dataset_label || "").toUpperCase() === labelFilter);
  }), [sampleRows, query, labelFilter]);
  const modelThreshold = Number(health?.threshold ?? 0.5);
  const resultProbability = Number(prediction?.fraud_probability ?? 0) * 100;
  const isOnline = health?.status === "healthy" || health?.status === "ok";

  function changeField(name, value) { setForm((old) => ({ ...old, [name]: value })); }
  function loadDemo(kind) {
    const demos = {
      transfer: { step: 1, type: "TRANSFER", amount: 850000, oldbalanceOrg: 850000, newbalanceOrig: 0, oldbalanceDest: 0, newbalanceDest: 850000 },
      everyday: { step: 24, type: "PAYMENT", amount: 2450, oldbalanceOrg: 12000, newbalanceOrig: 9550, oldbalanceDest: 3000, newbalanceDest: 5450 },
      cashout: { step: 18, type: "CASH_OUT", amount: 275000, oldbalanceOrg: 275000, newbalanceOrig: 0, oldbalanceDest: 12000, newbalanceDest: 287000 },
    };
    setForm(demos[kind]); setPrediction(null); setFormError(""); setPage("predict");
  }

  async function submitPrediction(e) {
    e.preventDefault(); setFormError(""); setPrediction(null);
    const keys = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"];
    for (const key of keys) {
      if (!Number.isFinite(Number(form[key])) || Number(form[key]) < 0) {
        setFormError(`${key} must be a valid non-negative number.`); return;
      }
    }
    setPredicting(true);
    try {
      const payload = { ...form };
      keys.forEach((key) => { payload[key] = Number(payload[key]); });
      const result = await predictFraud(payload);
      setPrediction(result);
    } catch (error) {
      setFormError(error?.response?.data?.detail || error?.message || "Prediction failed. Check that FastAPI and the model are running.");
    } finally { setPredicting(false); }
  }

  function exportDataset() {
    const columns = ["dataset_id", "step", "type", "amount", "dataset_label"];
    const escape = (v) => `"${String(v ?? "").replaceAll('"', '""')}"`;
    const csv = [columns.join(","), ...filteredRows.map((r, i) => columns.map((key) => escape(key === "dataset_id" ? (r.dataset_id || `PS-${i + 1}`) : r[key])).join(","))].join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8;" }));
    const link = document.createElement("a"); link.href = url; link.download = "paysim-preview.csv"; link.click(); URL.revokeObjectURL(url);
  }

  const pageTitle = {
    overview: ["FRAUDGUARD / OVERVIEW", "Welcome back, Vaishnavi.", "A real-data view of your fraud detection model and PaySim transaction patterns."],
    predict: ["RISK SCORING ENGINE", "Analyze a transaction", "Send transaction features to your deployed Random Forest model for a live assessment."],
    dataset: ["SOURCE DATA / PAYSIM", "Explore the PaySim dataset", "Browse source examples and ground-truth labels separately from model predictions."],
    analytics: ["DATA INSIGHTS", "Understand the data", "Explore the original PaySim dataset distribution and fraud activity by simulation step."],
  }[page];

  return <div className="app-shell">
    {mobileOpen && <button className="mobile-scrim" onClick={() => setMobileOpen(false)} aria-label="Close menu"/>}
    <aside className={`sidebar ${mobileOpen ? "sidebar-open" : ""}`}>
      <div className="brand"><div className="brand-mark"><ShieldCheck size={23}/></div><div><strong>Fraud<span>Guard</span></strong><small>BY VAISHNAVI KIRKAN</small></div><button className="mobile-close icon-button" onClick={() => setMobileOpen(false)} aria-label="Close menu"><X size={18}/></button></div>
      <div className="workspace-label">WORKSPACE</div>
      <nav className="side-nav">{NAV.map(({ id, label, icon: Icon }) => <button key={id} className={`nav-item ${page === id ? "active" : ""}`} onClick={() => { setPage(id); setMobileOpen(false); }}><Icon size={18}/><span>{label}</span>{id === "predict" && <span className="nav-arrow"><ArrowRight size={14}/></span>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="model-mini"><span className="pulse-dot"/><div><strong>{health?.model_name || "Fraud model"}</strong><small>{health?.model_loaded === false ? "Model not loaded" : isOnline ? "Inference service ready" : "Checking service"}</small></div></div><div className="profile-mini"><div className="avatar">VK</div><div><strong>Vaishnavi Kirkan</strong><small>Project owner</small></div></div></div>
    </aside>
    <main className="main-area">
      <header className="topbar"><button className="icon-button menu-trigger" onClick={() => setMobileOpen(true)} aria-label="Open menu"><Menu size={19}/></button><div className="breadcrumbs">Fraud Detection <span>/</span> <strong>{NAV.find((n) => n.id === page)?.label}</strong></div><div className="topbar-actions"><span className={`connection-pill ${isOnline ? "online" : health ? "warning" : "checking"}`}><i/>{isOnline ? "Model API ready" : health ? "Model unavailable" : "Connecting"}</span><button className="icon-button" title="Refresh live data" onClick={refresh}><RefreshCw size={17} className={refreshing ? "spin" : ""}/></button><button className="icon-button" title={`Switch to ${theme === "light" ? "dark" : "light"} mode`} onClick={() => setTheme(theme === "light" ? "dark" : "light")}>{theme === "light" ? <Moon size={18}/> : <Sun size={18}/>}</button></div></header>
      <div className="page-content">
        <div className="page-heading"><div><div className="eyebrow"><span className="eyebrow-dot"/>{pageTitle[0]}</div><h1>{pageTitle[1]}</h1><p>{pageTitle[2]}</p></div>{page === "overview" && <button className="primary-button" onClick={() => setPage("predict")}><ShieldCheck size={16}/> Check a transaction <ArrowRight size={15}/></button>}</div>
        {apiError && <div className="error-banner"><AlertTriangle size={17}/><span>{apiError} Confirm the backend is running at <code>127.0.0.1:8000</code>.</span><button onClick={refresh}>Retry</button></div>}

        {page === "overview" && <>
          <section className="hero-card"><div className="hero-copy"><span className="hero-chip"><Zap size={13}/> REAL-DATA FRAUD INTELLIGENCE</span><h2>Every transaction tells a story.<br/><em>Catch the unusual ones.</em></h2><p>Score a transaction in seconds, then explore the original PaySim patterns behind your fraud detection project.</p><button className="hero-button" onClick={() => setPage("predict")}>Run a risk assessment <ArrowRight size={16}/></button></div><div className="hero-visual"><div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="shield-orb"><ShieldCheck size={52}/></div><div className="floating-tag tag-top"><span className="mini-dot mint"/> Model inference</div><div className="floating-tag tag-bottom"><Activity size={14}/> Risk scoring</div></div></section>
          <div className="stats-grid"><Stat icon={Database} label="Total PaySim transactions" value={count(analytics?.total_transactions)} detail="Records in the source dataset" tone="blue"/><Stat icon={ShieldAlert} label="Source fraud cases" value={count(analytics?.fraud_transactions)} detail="Original dataset labels" tone="rose"/><Stat icon={CheckCircle2} label="Legitimate transactions" value={count(analytics?.legitimate_transactions)} detail="Original dataset labels" tone="mint"/><Stat icon={Gauge} label="Source fraud rate" value={pct(analytics?.fraud_rate_percent, 3)} detail="Highly imbalanced dataset" tone="amber"/></div>
          <div className="dashboard-grid"><section className="panel chart-panel"><SectionTitle kicker="PATTERN WATCH" title="Fraud activity over simulation steps" detail="Source labels by PaySim step — not model prediction history."/><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><AreaChart data={stepRows} margin={{ top: 12, right: 12, left: -18, bottom: 0 }}><defs><linearGradient id="fraudFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#ff6f91" stopOpacity={0.34}/><stop offset="100%" stopColor="#ff6f91" stopOpacity={0.015}/></linearGradient></defs><CartesianGrid stroke="var(--chart-grid)" strokeDasharray="4 5" vertical={false}/><XAxis dataKey="label" tickLine={false} axisLine={false} tick={{ fill: "var(--muted)", fontSize: 11 }} minTickGap={24}/><YAxis tickLine={false} axisLine={false} tick={{ fill: "var(--muted)", fontSize: 11 }}/><Tooltip contentStyle={{ background: "var(--tooltip-bg)", border: "1px solid var(--border)", borderRadius: 12, color: "var(--text)" }}/><Area type="monotone" dataKey="fraud_transactions" name="Source fraud cases" stroke="#ff6f91" strokeWidth={2.5} fill="url(#fraudFill)" activeDot={{ r: 5 }}/></AreaChart></ResponsiveContainer></div><div className="chart-footnote"><span><i className="legend-pink"/> Source-labelled fraud</span><span>{stepRows.length ? `${stepRows.length} step groups loaded` : "No chart data available"}</span></div></section>
            <section className="panel quick-panel"><SectionTitle kicker="TRY IT NOW" title="Quick risk check" detail="Load an example, then submit it to your model."/><button className="demo-option" onClick={() => loadDemo("transfer")}><span className="demo-icon rose"><ShieldAlert size={18}/></span><span><strong>Large transfer</strong><small>High-value balance movement</small></span><ArrowRight size={16}/></button><button className="demo-option" onClick={() => loadDemo("everyday")}><span className="demo-icon blue"><CreditCardIcon/></span><span><strong>Everyday payment</strong><small>Typical payment scenario</small></span><ArrowRight size={16}/></button><button className="demo-option" onClick={() => loadDemo("cashout")}><span className="demo-icon amber"><Activity size={18}/></span><span><strong>Cash-out scenario</strong><small>Balance changes worth checking</small></span><ArrowRight size={16}/></button><div className="quick-note"><Shield size={16}/><span>Examples are test inputs, not labelled fraud cases. The model decides the outcome.</span></div></section></div>
          <section className="panel source-panel"><div className="source-icon"><FileSpreadsheet size={20}/></div><div className="source-copy"><strong>Transparent by design</strong><p>Dataset ground truth and live model predictions are shown separately so you can tell what the model predicted versus what PaySim originally labelled.</p></div><button className="secondary-button" onClick={() => setPage("dataset")}>Explore source rows <ArrowRight size={15}/></button></section>
        </>}

        {page === "predict" && <div className="prediction-layout"><section className="panel form-panel"><SectionTitle kicker="MODEL INPUT" title="Transaction details" detail="All seven features expected by the deployed classifier."/><div className="demo-pills"><span>Try an example:</span><button onClick={() => loadDemo("transfer")}>Large transfer</button><button onClick={() => loadDemo("everyday")}>Everyday payment</button><button onClick={() => loadDemo("cashout")}>Cash-out</button></div><form onSubmit={submitPrediction}><label className="field"><span>Transaction type</span><select value={form.type} onChange={(e) => changeField("type", e.target.value)}>{TYPES.map((t) => <option key={t}>{t}</option>)}</select></label><div className="form-grid"><Field label="Simulation step" name="step" value={form.step} onChange={changeField} step="1"/><Field label="Transaction amount" name="amount" value={form.amount} onChange={changeField}/><Field label="Origin balance before" name="oldbalanceOrg" value={form.oldbalanceOrg} onChange={changeField}/><Field label="Origin balance after" name="newbalanceOrig" value={form.newbalanceOrig} onChange={changeField}/><Field label="Destination balance before" name="oldbalanceDest" value={form.oldbalanceDest} onChange={changeField}/><Field label="Destination balance after" name="newbalanceDest" value={form.newbalanceDest} onChange={changeField}/></div>{formError && <div className="form-error"><AlertTriangle size={16}/>{String(formError)}</div>}<div className="form-actions"><button type="button" className="secondary-button" onClick={() => { setForm(INITIAL_FORM); setPrediction(null); setFormError(""); }}>Reset form</button><button type="submit" className="primary-button" disabled={predicting}>{predicting ? <RefreshCw size={16} className="spin"/> : <ShieldCheck size={16}/>} {predicting ? "Analyzing…" : "Analyze transaction"} <ArrowRight size={15}/></button></div></form></section>
          <section className="panel result-panel"><SectionTitle kicker="LIVE MODEL RESPONSE" title="Risk assessment" detail="Result returned directly by the FastAPI prediction endpoint."/>{prediction ? <><div className={`result-status ${prediction.is_fraud ? "fraud" : "legit"}`}><span className="result-status-icon">{prediction.is_fraud ? <ShieldAlert size={24}/> : <ShieldCheck size={24}/>}</span><div><small>MODEL CLASSIFICATION</small><h3>{prediction.prediction || (prediction.is_fraud ? "FRAUD" : "LEGITIMATE")}</h3><p>{prediction.message || "Model response received."}</p></div></div><div className="risk-meter"><div className="risk-meter-top"><span>Fraud probability</span><strong>{pct(resultProbability, 2)}</strong></div><div className="risk-track"><div className={`risk-fill ${prediction.is_fraud ? "fraud" : "legit"}`} style={{ width: `${Math.max(0, Math.min(100, resultProbability))}%` }}/></div><div className="risk-meter-bottom"><span>0% · Low</span><span>100% · High</span></div></div><div className="result-details"><div><span>Decision threshold</span><strong>{pct(Number(prediction.threshold ?? modelThreshold) * 100, 1)}</strong></div><div><span>Model</span><strong>{prediction.model_name || health?.model_name || "Deployed classifier"}</strong></div><div><span>Transaction amount</span><strong>{money(form.amount)}</strong></div></div><div className="result-disclaimer"><AlertTriangle size={15}/> A model score is a decision-support signal, not a guarantee of fraud or legitimacy.</div><button className="secondary-button full-button" onClick={() => { setPrediction(null); setPage("predict"); }}>Analyze another transaction</button></> : <div className="result-placeholder"><div className="risk-ring"><Shield size={36}/></div><h3>Your result will appear here</h3><p>Enter transaction features or load an example, then run the model to see its classification, probability and threshold.</p><div className="placeholder-facts"><span><CheckCircle2 size={15}/> Live API inference</span><span><CheckCircle2 size={15}/> Probability + threshold</span><span><CheckCircle2 size={15}/> No fabricated history</span></div></div>}</section></div>}

        {page === "dataset" && <section className="panel dataset-panel"><SectionTitle kicker="GROUND-TRUTH SAMPLE" title="PaySim source records" detail={`${count(analytics?.total_transactions)} original records · showing a representative sample returned by your backend.`} action={<button className="secondary-button" onClick={exportDataset} disabled={!filteredRows.length}><Download size={15}/> Export filtered CSV</button>}/><div className="table-toolbar"><label className="search-box"><Search size={16}/><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search type, step, label or amount…"/></label><label className="filter-select"><ChevronDown size={15}/><select value={labelFilter} onChange={(e) => setLabelFilter(e.target.value)}><option value="ALL">All source labels</option><option value="FRAUD">Fraud</option><option value="LEGITIMATE">Legitimate</option></select></label></div><div className="table-scroll"><table className="data-table"><thead><tr><th>Sample ID</th><th>Step</th><th>Type</th><th>Amount</th><th>Original label</th><th>Action</th></tr></thead><tbody>{filteredRows.map((r, i) => <tr key={r.dataset_id || `${r.step}-${i}`}><td className="mono">{r.dataset_id || `PS-${i + 1}`}</td><td>{r.step ?? "—"}</td><td><span className="type-pill">{r.type || "—"}</span></td><td>{r.amount == null ? "—" : money(r.amount)}</td><td><span className={`label-pill ${outcomeIsFraud(r.dataset_label) ? "fraud" : "legit"}`}>{r.dataset_label || "Unknown"}</span></td><td><button className="table-action" onClick={() => { setForm({ ...INITIAL_FORM, step: Number(r.step || 1), type: TYPES.includes(r.type) ? r.type : "TRANSFER", amount: Number(r.amount || 0) }); setPrediction(null); setPage("predict"); }}>Score input <ArrowRight size={13}/></button></td></tr>)}</tbody></table></div>{!filteredRows.length && <Empty title="No matching source rows" detail={sampleRows.length ? "Try another search or label filter." : "The backend has not returned sample rows. Check the analytics endpoint and PaySim CSV path."}/>}<div className="table-note"><Shield size={15}/> These are PaySim ground-truth labels. They are not predictions produced by your saved model.</div></section>}

        {page === "analytics" && <><div className="stats-grid"><Stat icon={Database} label="Total records" value={count(analytics?.total_transactions)} detail="Full PaySim dataset"/><Stat icon={ShieldAlert} label="Fraud records" value={count(analytics?.fraud_transactions)} detail="Ground-truth positives" tone="rose"/><Stat icon={CheckCircle2} label="Legitimate records" value={count(analytics?.legitimate_transactions)} detail="Ground-truth negatives" tone="mint"/><Stat icon={Gauge} label="Fraud rate" value={pct(analytics?.fraud_rate_percent, 3)} detail="Original label distribution" tone="amber"/></div><div className="dashboard-grid"><section className="panel"><SectionTitle kicker="TRANSACTION MIX" title="Volume by transaction type" detail="Counts from the full source dataset."/><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><BarChart data={typeRows} margin={{ top: 10, right: 10, left: -14, bottom: 0 }}><CartesianGrid stroke="var(--chart-grid)" strokeDasharray="4 5" vertical={false}/><XAxis dataKey="name" tickLine={false} axisLine={false} tick={{ fill: "var(--muted)", fontSize: 10 }} interval={0}/><YAxis tickLine={false} axisLine={false} tick={{ fill: "var(--muted)", fontSize: 11 }}/><Tooltip contentStyle={{ background: "var(--tooltip-bg)", border: "1px solid var(--border)", borderRadius: 12, color: "var(--text)" }}/><Bar dataKey="value" name="Transactions" fill="#7d8dff" radius={[6, 6, 0, 0]}/></BarChart></ResponsiveContainer></div></section><section className="panel"><SectionTitle kicker="CLASS IMBALANCE" title="Original label distribution" detail="Why fraud detection needs more than accuracy."/><div className="donut-area"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={[{ name: "Legitimate", value: Number(analytics?.legitimate_transactions || 0), color: "#61cbb5" }, { name: "Fraud", value: Number(analytics?.fraud_transactions || 0), color: "#ff7898" }]} dataKey="value" nameKey="name" innerRadius="62%" outerRadius="84%" paddingAngle={3} stroke="none">{[{ color: "#61cbb5" }, { color: "#ff7898" }].map((x, i) => <Cell key={i} fill={x.color}/>)}</Pie><Tooltip contentStyle={{ background: "var(--tooltip-bg)", border: "1px solid var(--border)", borderRadius: 12, color: "var(--text)" }}/></PieChart></ResponsiveContainer><div className="donut-center"><strong>{pct(analytics?.fraud_rate_percent, 3)}</strong><span>fraud rate</span></div></div><div className="analytics-legend"><span><i className="legend-mint"/> Legitimate <b>{count(analytics?.legitimate_transactions)}</b></span><span><i className="legend-pink"/> Fraud <b>{count(analytics?.fraud_transactions)}</b></span></div></section></div><section className="panel insight-panel"><div className="insight-icon"><Activity size={20}/></div><div><div className="kicker">MODEL DEVELOPMENT NOTE</div><h3>Accuracy alone can hide missed fraud.</h3><p>Fraud is rare in the source dataset. When evaluating the model, inspect precision, recall, PR-AUC and false negatives alongside ROC-AUC, and tune the decision threshold for your use case.</p></div><button className="secondary-button" onClick={() => setPage("predict")}>Test a transaction <ArrowRight size={15}/></button></section></>}
        <footer className="app-footer"><span>FraudGuard <i/> PaySim · Fraud Detection MLOps</span><span>{health?.model_name ? `${health.model_name} · threshold ${Number(health.threshold ?? 0).toFixed(2)}` : "Waiting for model metadata"}</span></footer>
      </div>
    </main>
  </div>;
}

function CreditCardIcon() { return <ShieldCheck size={18}/>; }
