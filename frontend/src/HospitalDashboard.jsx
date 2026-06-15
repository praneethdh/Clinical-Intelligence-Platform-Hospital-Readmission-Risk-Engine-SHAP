import React, { useState } from 'react';

const RISK_CONFIG = {
  Critical: { color: '#dc2626', bg: 'rgba(220, 38, 38, 0.1)', border: 'rgba(220, 38, 38, 0.3)', icon: '🚨', label: 'CRITICAL RISK' },
  High:     { color: '#ef4444', bg: 'rgba(239, 68, 68, 0.1)',  border: 'rgba(239, 68, 68, 0.3)',  icon: '⚠️', label: 'HIGH RISK' },
  Elevated: { color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.1)', border: 'rgba(245, 158, 11, 0.3)', icon: '⚡', label: 'ELEVATED RISK' },
  Low:      { color: '#10b981', bg: 'rgba(16, 185, 129, 0.1)', border: 'rgba(16, 185, 129, 0.3)', icon: '✅', label: 'LOW RISK' },
};

const AGE_OPTIONS = [
  { value: '[0-10)', label: '0–10 Years' },
  { value: '[10-20)', label: '10–20 Years' },
  { value: '[20-30)', label: '20–30 Years' },
  { value: '[30-40)', label: '30–40 Years' },
  { value: '[40-50)', label: '40–50 Years' },
  { value: '[50-60)', label: '50–60 Years' },
  { value: '[60-70)', label: '60–70 Years' },
  { value: '[70-80)', label: '70–80 Years' },
  { value: '[80-90)', label: '80–90 Years' },
  { value: '[90-100)', label: '90–100 Years' },
];

const InputField = ({ label, icon, children }) => (
  <div className="space-y-2">
    <label className="flex items-center gap-2 text-sm font-medium text-slate-400">
      <span className="text-base">{icon}</span>
      {label}
    </label>
    {children}
  </div>
);

const inputClasses = "w-full px-4 py-3 rounded-xl bg-[#0f172a] border border-slate-700/50 text-slate-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all duration-300 placeholder:text-slate-600";

const parseBoldText = (text) => {
  if (!text) return null;
  const parts = text.split(/(\*\*.*?\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i} className="font-bold text-white">{part.slice(2, -2)}</strong>;
    }
    return part;
  });
};

const HospitalDashboard = () => {
  const [formData, setFormData] = useState({
    time_in_hospital: 5,
    num_lab_procedures: 40,
    num_procedures: 2,
    num_medications: 15,
    number_inpatient: 0,
    number_outpatient: 0,
    number_emergency: 0,
    number_diagnoses: 5,
    discharge_disposition_id: 1,
    admission_type_id: 1,
    admission_source_id: 7,
    gender: 'Female',
    race: 'Caucasian',
    age: '[70-80)',
    diabetesMed_Yes: 1,
    insulin: 'No',
    change: 'No',
    diag_1: '434',
    payer_code: 'MC',
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Prediction failed');
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err.message === 'Failed to fetch'
          ? 'Cannot reach the API. Start the backend: cd backend && uvicorn main:app --reload --port 8000'
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const riskCfg = result ? RISK_CONFIG[result.risk_level] || RISK_CONFIG.Low : null;

  return (
    <div className="min-h-screen relative overflow-hidden" style={{ background: '#0f172a' }}>
      {/* Floating gradient orbs */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="float-orb absolute -top-32 -left-32 w-96 h-96 rounded-full opacity-20" style={{ background: 'radial-gradient(circle, #3b82f6, transparent)' }} />
        <div className="float-orb absolute top-1/2 -right-48 w-[30rem] h-[30rem] rounded-full opacity-10" style={{ background: 'radial-gradient(circle, #8b5cf6, transparent)', animationDelay: '2s' }} />
        <div className="float-orb absolute -bottom-24 left-1/3 w-80 h-80 rounded-full opacity-15" style={{ background: 'radial-gradient(circle, #06b6d4, transparent)', animationDelay: '4s' }} />
      </div>

      <div className="relative z-10 max-w-6xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {/* Header */}
        <header className="text-center mb-10">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold tracking-wide uppercase mb-4"
               style={{ background: 'rgba(59, 130, 246, 0.15)', color: '#60a5fa', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
            <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
            AI-Powered Clinical Intelligence
          </div>
          <h1 className="text-4xl sm:text-5xl font-black tracking-tight text-white mb-3">
            Readmission Risk
            <span className="bg-gradient-to-r from-blue-400 via-violet-400 to-cyan-400 bg-clip-text text-transparent"> Engine</span>
          </h1>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            Predict 30-day hospital readmission probability with XGBoost + SHAP explainability
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Left Panel – Patient Form */}
          <div className="lg:col-span-3 glass-card rounded-2xl p-6 sm:p-8">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center text-xl"
                   style={{ background: 'rgba(59, 130, 246, 0.15)' }}>🏥</div>
              <div>
                <h2 className="text-lg font-bold text-white">Patient Data Entry</h2>
                <p className="text-xs text-slate-500">Enter clinical parameters for risk assessment</p>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
              {/* Row 1: Demographics */}
              <InputField label="Age Bracket" icon="👤">
                <select className={inputClasses} value={formData.age} onChange={e => handleChange('age', e.target.value)}>
                  {AGE_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </InputField>

              <InputField label="Gender" icon="⚧">
                <select className={inputClasses} value={formData.gender} onChange={e => handleChange('gender', e.target.value)}>
                  <option value="Female">Female</option>
                  <option value="Male">Male</option>
                  <option value="Other">Other</option>
                </select>
              </InputField>

              {/* Row 2: Diagnosis */}
              <InputField label="Primary Diagnosis (ICD-9)" icon="🩺">
                <input type="text" className={inputClasses} placeholder="e.g., 434, 250, 428"
                       value={formData.diag_1} onChange={e => handleChange('diag_1', e.target.value)} />
              </InputField>

              <InputField label="Days in Hospital" icon="🛏️">
                <input type="number" min="1" max="14" className={inputClasses}
                       value={formData.time_in_hospital} onChange={e => handleChange('time_in_hospital', parseInt(e.target.value) || 0)} />
              </InputField>

              {/* Row 3: Procedures */}
              <InputField label="Lab Procedures" icon="🔬">
                <input type="number" min="0" className={inputClasses}
                       value={formData.num_lab_procedures} onChange={e => handleChange('num_lab_procedures', parseInt(e.target.value) || 0)} />
              </InputField>

              <InputField label="Medications Count" icon="💊">
                <input type="number" min="0" className={inputClasses}
                       value={formData.num_medications} onChange={e => handleChange('num_medications', parseInt(e.target.value) || 0)} />
              </InputField>

              {/* Row 4: History */}
              <InputField label="Prior Inpatient Visits" icon="📋">
                <input type="number" min="0" className={inputClasses}
                       value={formData.number_inpatient} onChange={e => handleChange('number_inpatient', parseInt(e.target.value) || 0)} />
              </InputField>

              <InputField label="Emergency Visits" icon="🚑">
                <input type="number" min="0" className={inputClasses}
                       value={formData.number_emergency} onChange={e => handleChange('number_emergency', parseInt(e.target.value) || 0)} />
              </InputField>

              {/* Row 5: Admin */}
              <InputField label="Discharge Disposition ID" icon="🏠">
                <input type="number" min="1" className={inputClasses}
                       value={formData.discharge_disposition_id} onChange={e => handleChange('discharge_disposition_id', parseInt(e.target.value) || 1)} />
              </InputField>

              <InputField label="Diabetes Medication" icon="💉">
                <select className={inputClasses} value={formData.diabetesMed_Yes} onChange={e => handleChange('diabetesMed_Yes', parseInt(e.target.value))}>
                  <option value={1}>Yes</option>
                  <option value={0}>No</option>
                </select>
              </InputField>
            </div>

            {/* Predict Button */}
            <button
              onClick={handlePredict}
              disabled={loading}
              className="w-full mt-8 py-4 rounded-xl font-bold text-white text-base tracking-wide transition-all duration-300 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed relative overflow-hidden group"
              style={{
                background: loading ? '#334155' : 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
                boxShadow: loading ? 'none' : '0 0 30px rgba(59, 130, 246, 0.3)',
              }}
            >
              <span className="relative z-10 flex items-center justify-center gap-2">
                {loading ? (
                  <>
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
                    Analyzing Patient Data…
                  </>
                ) : (
                  <>
                    <span className="text-lg">⚡</span>
                    Run Risk Assessment
                  </>
                )}
              </span>
              {!loading && (
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-700" />
              )}
            </button>

            {error && (
              <div className="mt-4 p-4 rounded-xl text-sm font-medium fade-in-up"
                   style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', color: '#fca5a5' }}>
                ❌ {error}
              </div>
            )}
          </div>

          {/* Right Panel – Results */}
          <div className="lg:col-span-2 space-y-6">
            {/* Risk Score Card */}
            <div className="glass-card rounded-2xl p-6 sm:p-8 min-h-[240px] flex flex-col justify-center">
              {!result && !loading && (
                <div className="text-center py-8">
                  <div className="text-5xl mb-4 opacity-30">🔬</div>
                  <p className="text-slate-500 text-sm">Enter patient data and run the assessment to see results</p>
                </div>
              )}

              {loading && (
                <div className="space-y-4">
                  <div className="h-8 rounded-lg shimmer" />
                  <div className="h-20 rounded-lg shimmer" />
                  <div className="h-6 rounded-lg shimmer w-2/3" />
                </div>
              )}

              {result && (
                <div className="fade-in-up">
                  {/* Risk Level Badge */}
                  <div className="flex items-center gap-2 mb-5">
                    <span className={`text-2xl ${result.risk_level === 'Critical' || result.risk_level === 'High' ? 'risk-pulse' : ''}`}>
                      {riskCfg.icon}
                    </span>
                    <span className="px-3 py-1 rounded-full text-xs font-bold tracking-widest uppercase"
                          style={{ background: riskCfg.bg, color: riskCfg.color, border: `1px solid ${riskCfg.border}` }}>
                      {riskCfg.label}
                    </span>
                  </div>

                  {/* Big Score */}
                  <div className="mb-5">
                    <div className="text-6xl font-black tracking-tight" style={{ color: riskCfg.color }}>
                      {result.probability_score}
                      <span className="text-2xl font-bold text-slate-500">%</span>
                    </div>
                    <p className="text-xs text-slate-500 mt-1 uppercase tracking-wider">Readmission Probability</p>
                  </div>

                  {/* Progress Bar */}
                  <div className="w-full h-2.5 rounded-full overflow-hidden mb-5" style={{ background: 'rgba(255,255,255,0.05)' }}>
                    <div className="h-full rounded-full animate-bar transition-all duration-700"
                         style={{ width: `${Math.min(result.probability_score, 100)}%`, background: `linear-gradient(90deg, ${riskCfg.color}, ${riskCfg.color}88)` }} />
                  </div>

                  {/* Recommendation */}
                  <div className="p-4 rounded-xl text-sm" style={{ background: riskCfg.bg, border: `1px solid ${riskCfg.border}` }}>
                    <p className="font-semibold mb-1" style={{ color: riskCfg.color }}>Clinical Recommendation</p>
                    <p className="text-slate-300 text-xs leading-relaxed">{result.recommendation}</p>
                  </div>
                </div>
              )}
            </div>

            {/* SHAP Reasoning Card */}
            {result && result.reasoning && result.reasoning.length > 0 && (
              <div className="glass-card rounded-2xl p-6 sm:p-8 fade-in-up" style={{ animationDelay: '0.2s' }}>
                <div className="flex items-center gap-3 mb-5">
                  <div className="w-9 h-9 rounded-lg flex items-center justify-center text-lg"
                       style={{ background: 'rgba(139, 92, 246, 0.15)' }}>🧠</div>
                  <div>
                    <h3 className="text-base font-bold text-white">AI Reasoning</h3>
                    <p className="text-xs text-slate-500">SHAP-powered feature attribution</p>
                  </div>
                </div>

                {result.reasoning_summary && (
                  <div className="p-4 rounded-xl text-sm mb-6 bg-slate-800/40 border border-slate-700/30 text-slate-300 leading-relaxed">
                    <p className="font-semibold text-violet-400 mb-1 text-xs uppercase tracking-wider flex items-center gap-1.5">
                      <span>📝</span> Clinical Risk Explanation
                    </p>
                    <p>{parseBoldText(result.reasoning_summary)}</p>
                  </div>
                )}

                <div className="space-y-4">
                  {result.reasoning.map((r, i) => {
                    const isPositive = r.direction === 'increases';
                    const barColor = isPositive ? '#ef4444' : '#10b981';
                    const maxImpact = Math.max(...result.reasoning.map(x => x.impact));
                    const barWidth = maxImpact > 0 ? (r.impact / maxImpact) * 100 : 0;

                    return (
                      <div key={i} className="fade-in-up" style={{ animationDelay: `${0.1 * i}s` }}>
                        <div className="flex items-center justify-between mb-1.5">
                          <span className="text-sm font-medium text-slate-300 truncate mr-2">{r.feature}</span>
                          <span className="text-xs font-mono px-2 py-0.5 rounded-md shrink-0"
                                style={{ background: isPositive ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.15)',
                                         color: barColor }}>
                            {isPositive ? '↑' : '↓'} {r.impact.toFixed(3)}
                          </span>
                        </div>
                        <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.05)' }}>
                          <div className="h-full rounded-full animate-bar"
                               style={{ width: `${barWidth}%`, background: barColor, animationDelay: `${0.15 * i}s` }} />
                        </div>
                        <p className="text-xs text-slate-500 mt-1">{r.explanation}</p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-12 pb-6">
          <p className="text-xs text-slate-600">
            Built with XGBoost · SHAP · FastAPI · React — Model F1-Optimized for Patient Safety
          </p>
        </footer>
      </div>
    </div>
  );
};

export default HospitalDashboard;
