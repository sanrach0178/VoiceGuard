"use client";
import React, { useState, useEffect, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { AlertTriangle, ShieldCheck, ShieldAlert, Activity, PhoneCall } from "lucide-react";

export default function Dashboard() {
  const [isRecording, setIsRecording] = useState(false);
  const [riskScore, setRiskScore] = useState(0);
  const [anomalyType, setAnomalyType] = useState("None");
  const [actionRecommended, setActionRecommended] = useState("Allow");
  const [history, setHistory] = useState<{ time: string; risk: number }[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [wsError, setWsError] = useState("");
  
  const [durationSecs, setDurationSecs] = useState(0);
  const [noiseLevel, setNoiseLevel] = useState("Low");
  const [audioData, setAudioData] = useState<number[]>(new Array(120).fill(10));
  const [spectrogramData, setSpectrogramData] = useState<number[]>(new Array(100).fill(0));

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationRef = useRef<number | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const durationIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const formatDuration = (secs: number) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `00:${m}:${s}`;
  };

  const connectWebSocket = () => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
        return;
    }

    const ws = new WebSocket("ws://localhost:8000/ws/stream");

    ws.onopen = () => {
      setIsConnected(true);
      setWsError("");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const score = data.risk_score || 0;
        const action = data.action || "Allow";
        
        let anomaly = "None";
        if (data.details && data.details.prosody) {
          const { pitch_std, jitter, silence_ratio } = data.details.prosody;
          const anomalies = [];
          if (pitch_std < 5.0 && pitch_std > 0) anomalies.push("Flat Pitch");
          if (jitter > 0.05) anomalies.push("High Jitter");
          if (silence_ratio < 0.005) anomalies.push("No Micro-breaths");
          if (anomalies.length > 0) {
             anomaly = anomalies.join(" | ");
          } else {
             if (score >= 50) anomaly = "Acoustic Discrepancy";
          }
        } else if (data.anomaly_type) {
           anomaly = data.anomaly_type;
        }

        if (data.details && data.details.rms_volume !== undefined) {
           const rms = data.details.rms_volume;
           if (rms < 0.01) setNoiseLevel("Quiet");
           else if (rms < 0.05) setNoiseLevel("Low");
           else if (rms < 0.15) setNoiseLevel("Medium");
           else setNoiseLevel("Loud");
        }

        setRiskScore(score);
        setAnomalyType(anomaly);
        setActionRecommended(action);

        const timeString = new Date(data.timestamp * 1000).toLocaleTimeString([], { hour12: false });
        setHistory((prev) => {
          const newHistory = [...prev, { time: timeString, risk: score }];
          // 2 updates per sec. 20 seconds = 40 points.
          if (newHistory.length > 40) return newHistory.slice(newHistory.length - 40);
          return newHistory;
        });
      } catch (err) {
        console.error("Error parsing message", err);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      wsRef.current = null;
      reconnectTimeoutRef.current = setTimeout(() => {
        connectWebSocket();
      }, 3000);
    };

    ws.onerror = () => {
      setWsError("WebSocket Error: Backend might be down.");
    };

    wsRef.current = ws;
  };

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
      stopRecording();
    };
  }, []);

  const convertFloat32ToInt16 = (buffer: Float32Array) => {
    let l = buffer.length;
    const buf = new Int16Array(l);
    while (l--) {
      let s = Math.max(-1, Math.min(1, buffer[l]));
      buf[l] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return buf.buffer;
  };

  const updateVisualizer = () => {
    if (!analyserRef.current) return;
    
    const timeData = new Uint8Array(analyserRef.current.fftSize);
    analyserRef.current.getByteTimeDomainData(timeData);
    
    const bars = [];
    const step = Math.floor(timeData.length / 120);
    for (let i = 0; i < 120; i++) {
       let sum = 0;
       for (let j = 0; j < step; j++) {
          sum += Math.abs(timeData[i * step + j] - 128);
       }
       const avg = sum / step;
       bars.push(Math.max(4, (avg / 128) * 100 * 3));
    }
    setAudioData(bars);

    const freqData = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(freqData);
    
    const specBars = [];
    const specStep = Math.floor(freqData.length / 100);
    for (let i = 0; i < 100; i++) {
       let sum = 0;
       for (let j = 0; j < specStep; j++) {
          sum += freqData[i * specStep + j];
       }
       specBars.push(sum / specStep / 255);
    }
    setSpectrogramData(specBars);

    animationRef.current = requestAnimationFrame(updateVisualizer);
  };

  const startRecording = async () => {
    try {
      if (!isConnected) {
        connectWebSocket();
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        }
      });
      mediaStreamRef.current = stream;

      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000,
      });
      audioContextRef.current = audioContext;

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 512;
      analyserRef.current = analyser;
      
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      
      updateVisualizer();

      const processor = audioContext.createScriptProcessor(8192, 1, 1);
      scriptProcessorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        const pcmData = convertFloat32ToInt16(inputData);
        
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(pcmData);
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      setIsRecording(true);
      setDurationSecs(0);
      durationIntervalRef.current = setInterval(() => {
        setDurationSecs(prev => prev + 1);
      }, 1000);
      
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Could not access microphone. Please ensure permissions are granted.");
    }
  };

  const stopRecording = () => {
    if (scriptProcessorRef.current) scriptProcessorRef.current.disconnect();
    if (audioContextRef.current) audioContextRef.current.close();
    if (mediaStreamRef.current) mediaStreamRef.current.getTracks().forEach((track) => track.stop());
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    if (durationIntervalRef.current) clearInterval(durationIntervalRef.current);
    
    setIsRecording(false);
    setAudioData(new Array(120).fill(4));
    setSpectrogramData(new Array(100).fill(0));
    setDurationSecs(0);
    setNoiseLevel("N/A");
  };

  const isHighRisk = riskScore > 70;

  const renderGauge = (score: number) => {
    const radius = 90;
    const circumference = Math.PI * radius;
    const offset = circumference - (score / 100) * circumference;
    const color = score > 70 ? '#ef4444' : score > 30 ? '#f59e0b' : '#22c55e';
    
    return (
      <div className="relative w-64 h-36 mx-auto mt-4 overflow-hidden flex justify-center">
        <svg className="absolute w-full h-full" viewBox="0 0 200 120">
          <path d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke="#f1f5f9" strokeWidth="12" strokeLinecap="round" />
          <path 
            d="M 10 100 A 90 90 0 0 1 190 100" 
            fill="none" 
            stroke={color} 
            strokeWidth="12" 
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        <div className="absolute bottom-2 flex flex-col items-center">
          <div className="text-6xl font-bold text-slate-800 tracking-tight">{score.toFixed(0)}<span className="text-3xl">%</span></div>
          <div className={`font-semibold text-sm mt-1 ${score > 70 ? 'text-red-600' : score > 30 ? 'text-amber-600' : 'text-green-500'}`}>
            {score > 70 ? 'High Risk Detected' : score > 30 ? 'Suspicious Audio' : 'No Risk Detected'}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans p-6 pb-20">
      <header className="flex justify-between items-center bg-white p-4 rounded-xl shadow-sm border border-slate-100 mb-6">
        <div className="flex items-center gap-3">
          <div className="text-blue-600 border border-blue-200 p-1.5 rounded-lg">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900 leading-none mb-0.5">VoiceGuard</h1>
            <p className="text-[10px] text-slate-400 font-semibold uppercase tracking-widest">AI Voice Fraud Detection</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <span className="text-slate-400 font-mono text-xs font-semibold">#CALL-884</span>
          <div className={`px-3 py-1.5 rounded-md text-xs font-bold flex items-center gap-2 ${isConnected ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-600'}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></span>
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>
        </div>
      </header>
      
      {wsError && !isConnected && (
        <div className="bg-red-50 border border-red-100 text-red-600 p-3 rounded-lg mb-6 text-sm flex justify-between items-center shadow-sm">
          <div className="flex items-center gap-3">
             <Activity className="w-4 h-4" />
             <span className="font-medium">{wsError}</span>
          </div>
          <button className="text-red-400 hover:text-red-600" onClick={() => setWsError("")}>✕</button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-white rounded-xl p-6 border border-slate-100 shadow-[0_2px_10px_-3px_rgba(6,81,237,0.05)]">
            <h2 className="text-xs text-slate-400 font-bold mb-4 uppercase tracking-wider text-center">Risk Score</h2>
            {renderGauge(riskScore)}
            <div className="mt-8 flex flex-col items-center text-center px-4">
              <div className="flex items-center gap-2 mb-2">
                {isHighRisk ? <ShieldAlert className="text-red-600 w-5 h-5" /> : <ShieldCheck className="text-green-500 w-5 h-5" />}
                <span className={`font-semibold text-sm ${isHighRisk ? 'text-red-600' : 'text-green-600'}`}>
                  {isHighRisk ? "Synthetic Voice Detected" : "Authentic Voice"}
                </span>
              </div>
              <p className="text-xs text-slate-500 font-medium">
                {isHighRisk ? "High confidence of AI generation in audio stream. Verify caller identity." : "The voice appears to be authentic. No anomalies detected in real-time analysis."}
              </p>
            </div>
          </div>

          <div className="bg-white rounded-xl p-6 border border-slate-100 shadow-[0_2px_10px_-3px_rgba(6,81,237,0.05)]">
            <h2 className="text-xs text-slate-500 font-bold mb-6 uppercase tracking-wider">Analysis Controls</h2>
            
            <div className="space-y-4">
              <div className="flex justify-between items-center py-2 border-b border-slate-50">
                <span className="text-sm text-slate-600 font-medium">Anomaly Detection</span>
                <div className="flex items-center gap-1 text-sm font-semibold text-blue-600">
                  {anomalyType} <Activity className="w-4 h-4 ml-1 opacity-50" />
                </div>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-slate-50">
                <span className="text-sm text-slate-600 font-medium">Action on Risk</span>
                <div className="flex items-center gap-1 text-sm font-semibold text-blue-600">
                  {actionRecommended} <Activity className="w-4 h-4 ml-1 opacity-50" />
                </div>
              </div>
            </div>

            {/* Mic Input */}
            <div className="mt-6 flex items-center gap-4">
              <div className="w-20">
                 <span className="text-xs text-slate-500 font-medium">Mic Input</span>
              </div>
              <Activity className="w-4 h-4 text-green-500" />
              <div className="flex-1 flex gap-1 h-2">
                 {Array.from({length: 15}).map((_, i) => {
                    // Calculate active bars based on noiseLevel string
                    let activeBars = 0;
                    if (noiseLevel === "Quiet") activeBars = 2;
                    else if (noiseLevel === "Low") activeBars = 5;
                    else if (noiseLevel === "Medium") activeBars = 10;
                    else if (noiseLevel === "Loud") activeBars = 15;
                    
                    if (!isRecording) activeBars = 0;
                    
                    return (
                      <div key={i} className={`flex-1 rounded-sm ${i < activeBars ? 'bg-green-500' : 'bg-slate-100'}`}></div>
                    );
                 })}
              </div>
              <span className="text-xs text-slate-400 font-medium w-10 text-right">{isRecording ? 'Live' : 'Muted'}</span>
            </div>

            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`w-full mt-8 py-3 rounded-lg font-bold flex justify-center items-center gap-2 transition-colors text-sm shadow-sm ${
                isRecording 
                ? 'bg-red-50 text-red-600 hover:bg-red-100' 
                : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              <Activity className="w-4 h-4" />
              {isRecording ? "Stop Call Monitoring" : "Start Live Monitoring"}
            </button>
          </div>
        </div>

        <div className="lg:col-span-8 flex flex-col gap-6">
          <div className="bg-white rounded-xl p-6 border border-slate-100 shadow-[0_2px_10px_-3px_rgba(6,81,237,0.05)]">
            <div className="flex justify-between items-start mb-2">
              <div className="flex items-center gap-3">
                <h2 className="text-xs text-slate-700 font-bold uppercase tracking-wider">Live Voice Monitor</h2>
                {isRecording && (
                   <span className="flex items-center gap-1 text-xs font-semibold text-green-500">
                     <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span> Live
                   </span>
                )}
              </div>
              <div className="flex gap-8 text-xs">
                <div>
                  <div className="text-slate-400 mb-1">Noise Level</div>
                  <div className="font-semibold text-slate-700 flex items-center gap-2">{noiseLevel} <div className="flex gap-0.5"><div className="w-1 h-1 bg-blue-400 rounded-full"></div><div className="w-1 h-1 bg-blue-400 rounded-full"></div><div className="w-1 h-1 bg-blue-400 rounded-full"></div><div className={`w-1 h-1 ${noiseLevel === 'Loud' ? 'bg-blue-400' : 'bg-blue-100'} rounded-full`}></div><div className="w-1 h-1 bg-blue-100 rounded-full"></div></div></div>
                </div>
                <div>
                  <div className="text-slate-400 mb-1">Input Source</div>
                  <div className="font-semibold text-slate-700 flex items-center gap-1"><Activity className="w-3 h-3 text-blue-500"/> Mic</div>
                </div>
                <div>
                  <div className="text-slate-400 mb-1">Duration</div>
                  <div className="font-semibold text-slate-700 flex items-center gap-1"><Activity className="w-3 h-3 text-blue-500"/> {formatDuration(durationSecs)}</div>
                </div>
              </div>
            </div>
            
            <div className="w-full h-32 flex items-center justify-center gap-[2px] mt-6 mb-2">
               {audioData.map((height, i) => (
                  <div key={i} className="w-1 bg-blue-400 rounded-full" style={{ height: `${height}%` }}></div>
               ))}
            </div>
            
            <div className="w-full h-16 mt-2 rounded overflow-hidden flex items-end bg-slate-50">
               {spectrogramData.map((val, i) => (
                  <div key={i} className="flex-1" style={{ height: '100%', opacity: val, backgroundColor: '#3b82f6' }}></div>
               ))}
            </div>
          </div>

          <div className="bg-white rounded-xl p-6 border border-slate-100 shadow-[0_2px_10px_-3px_rgba(6,81,237,0.05)] flex-1 min-h-[300px] flex flex-col">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xs text-slate-700 font-bold uppercase tracking-wider">Risk History (20s)</h2>
            </div>
            
            <div className="flex-1 w-full h-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history} margin={{ top: 5, right: 10, bottom: 5, left: -25 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis dataKey="time" stroke="#94a3b8" fontSize={10} tickMargin={12} tickLine={false} axisLine={false} />
                  <YAxis stroke="#94a3b8" fontSize={10} domain={[0, 100]} tickCount={5} tickLine={false} axisLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)', padding: '12px' }}
                    itemStyle={{ color: '#22c55e', fontWeight: '700' }}
                    labelStyle={{ color: '#64748b', fontSize: '11px', marginBottom: '4px' }}
                  />
                  <Line type="monotone" dataKey="risk" stroke="#22c55e" strokeWidth={2} dot={{ r: 4, fill: '#22c55e', strokeWidth: 0 }} activeDot={{ r: 6, fill: '#22c55e', strokeWidth: 0 }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 bg-white rounded-xl border border-slate-100 shadow-[0_2px_10px_-3px_rgba(6,81,237,0.05)] p-4 px-8 flex items-center">
        <h2 className="text-xs text-slate-700 font-bold uppercase tracking-wider w-40">System Status</h2>
        
        <div className="flex-1 flex justify-between items-center px-10 border-l border-slate-100">
          <div className="flex items-center gap-3">
             <div className="w-8 h-8 rounded-full bg-green-50 flex items-center justify-center border border-green-100">
               <ShieldCheck className="w-4 h-4 text-green-500" />
             </div>
             <div>
               <div className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider">Model Status</div>
               <div className="text-xs font-bold text-green-600">Healthy</div>
             </div>
          </div>
          
          <div className="flex items-center gap-3">
             <div className="w-8 h-8 rounded-full bg-blue-50 flex items-center justify-center border border-blue-100">
               <Activity className="w-4 h-4 text-blue-500" />
             </div>
             <div>
               <div className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider">Data Stream</div>
               <div className="text-xs font-bold text-green-600">{isRecording ? 'Active' : 'Standby'}</div>
             </div>
          </div>
          
          <div className="flex items-center gap-3">
             <div className="w-8 h-8 rounded-full bg-blue-50 flex items-center justify-center border border-blue-100">
               <Activity className="w-4 h-4 text-blue-500" />
             </div>
             <div>
               <div className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider">Connection</div>
               <div className={`text-xs font-bold ${isConnected ? 'text-green-600' : 'text-amber-500'}`}>
                 {isConnected ? 'Stable' : 'Reconnecting...'}
               </div>
             </div>
          </div>

          <div className="flex items-center gap-3">
             <div className="w-8 h-8 rounded-full bg-blue-50 flex items-center justify-center border border-blue-100">
               <Activity className="w-4 h-4 text-blue-500" />
             </div>
             <div>
               <div className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider">Uptime</div>
               <div className="text-xs font-bold text-blue-600">{formatDuration(durationSecs)}</div>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
}
