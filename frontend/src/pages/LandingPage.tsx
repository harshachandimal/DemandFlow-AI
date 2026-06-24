import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, BarChart3, TrendingUp, FileText, ArrowRight, Database, Server, Cpu } from 'lucide-react';
import Button from '../components/ui/Button';
import Card from '../components/ui/Card';

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-[#0b0f1a] text-slate-200 selection:bg-cyan-500/30 overflow-hidden relative font-sans">
      
      {/* Background Ambience */}
      <div className="absolute inset-0 z-0 pointer-events-none" aria-hidden="true">
        <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] rounded-full bg-cyan-900/20 blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-emerald-900/10 blur-[120px]" />
      </div>

      <div className="relative z-10 max-w-6xl mx-auto px-6 pt-24 pb-32">
        
        {/* Header / Hero Section */}
        <header className="text-center mb-24 fade-up">
          <div className="inline-flex items-center justify-center p-3 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-emerald-500/20 border border-cyan-500/20 mb-8 shadow-[0_0_30px_rgba(6,182,212,0.15)]">
            <Activity size={32} className="text-cyan-400" />
          </div>
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6">
            DemandFlow <span className="gradient-text">AI</span>
          </h1>
          <p className="text-xl md:text-2xl text-slate-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            Enterprise-grade demand forecasting powered by deep learning. Anticipate sales, optimize inventory, and mitigate risk with precision.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Button 
              className="text-lg px-8 py-4 rounded-xl"
              onClick={() => navigate('/dashboard')}
              icon={<ArrowRight size={20} />}
            >
              Open Dashboard
            </Button>
            <Button 
              variant="ghost" 
              className="text-lg px-8 py-4 rounded-xl"
              onClick={() => {
                document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' });
              }}
            >
              Explore Features
            </Button>
          </div>
        </header>

        {/* Feature Highlights */}
        <section id="features" className="mb-32">
          <div className="text-center mb-12">
            <h2 className="text-xs font-bold text-slate-500 tracking-widest uppercase mb-2">Capabilities</h2>
            <h3 className="text-3xl font-bold text-slate-200">Everything you need to predict demand</h3>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <FeatureCard 
              icon={<BarChart3 size={24} className="text-cyan-400" />}
              title="AI Demand Forecasting"
              description="7-day lookahead predictions utilizing LSTM neural networks trained on historical sales."
            />
            <FeatureCard 
              icon={<AlertIcon />}
              title="Inventory Risk Detection"
              description="Automated alerts for stockouts and overstock based on real-time predictions."
            />
            <FeatureCard 
              icon={<TrendingUp size={24} className="text-emerald-400" />}
              title="Scenario Planning"
              description="A/B test different pricing and promotional strategies to see impact on demand."
            />
            <FeatureCard 
              icon={<FileText size={24} className="text-indigo-400" />}
              title="Executive Reports"
              description="Export rich, presentation-ready insights as PDF, CSV, or JSON in one click."
            />
          </div>
        </section>

        {/* Model Credibility & Architecture */}
        <section className="grid lg:grid-cols-2 gap-12 items-center">
          
          {/* Model Stats */}
          <div className="fade-up" style={{ animationDelay: '0.1s' }}>
            <h2 className="text-xs font-bold text-slate-500 tracking-widest uppercase mb-2">Performance</h2>
            <h3 className="text-3xl font-bold text-slate-200 mb-6">Proven Model Accuracy</h3>
            <p className="text-slate-400 mb-8 leading-relaxed">
              Our core forecasting engine is built on a customized Long Short-Term Memory (LSTM) architecture, fine-tuned on the Rossmann dataset to capture complex seasonal patterns, promotions, and holiday impacts.
            </p>
            
            <div className="grid grid-cols-2 gap-4">
              <StatCard label="Model Version" value="Rossmann LSTM v2" />
              <StatCard label="MAPE" value="7.32%" />
              <StatCard label="MAE" value="332.43" />
              <StatCard label="RMSE" value="439.03" />
            </div>
          </div>

          {/* Architecture Flow */}
          <div className="fade-up" style={{ animationDelay: '0.2s' }}>
            <Card className="p-8 border-slate-700/50 bg-slate-800/20">
              <h3 className="text-lg font-semibold text-slate-300 mb-6 flex items-center gap-2">
                <Database size={20} className="text-cyan-500" />
                System Architecture
              </h3>
              
              <div className="space-y-4">
                <ArchitectureStep 
                  icon={<Activity size={20} className="text-cyan-400" />}
                  title="React Frontend"
                  desc="Interactive UI / Glassmorphism Design"
                  isLast={false}
                />
                <ArchitectureStep 
                  icon={<Server size={20} className="text-indigo-400" />}
                  title="Laravel Backend"
                  desc="PHP 8 API / Database & State Management"
                  isLast={false}
                />
                <ArchitectureStep 
                  icon={<Cpu size={20} className="text-emerald-400" />}
                  title="FastAPI Microservice"
                  desc="Python / Model Serving & Preprocessing"
                  isLast={false}
                />
                <ArchitectureStep 
                  icon={<BarChart3 size={20} className="text-rose-400" />}
                  title="PyTorch LSTM"
                  desc="Deep Learning Model / Inference"
                  isLast={true}
                />
              </div>
            </Card>
          </div>

        </section>

      </div>
    </div>
  );
}

// Subcomponents

function FeatureCard({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) {
  return (
    <Card className="hover:-translate-y-1 transition-transform duration-300 cursor-default group">
      <div className="p-3 bg-slate-800/50 rounded-xl inline-block mb-4 border border-slate-700/50 group-hover:border-cyan-500/30 transition-colors">
        {icon}
      </div>
      <h4 className="text-lg font-semibold text-slate-200 mb-2">{title}</h4>
      <p className="text-sm text-slate-400 leading-relaxed">{description}</p>
    </Card>
  );
}

function StatCard({ label, value }: { label: string, value: string }) {
  return (
    <div className="bg-slate-800/30 border border-slate-700/50 rounded-xl p-5">
      <div className="text-sm text-slate-400 mb-1">{label}</div>
      <div className="text-xl font-bold text-slate-100">{value}</div>
    </div>
  );
}

function ArchitectureStep({ icon, title, desc, isLast }: { icon: React.ReactNode, title: string, desc: string, isLast: boolean }) {
  return (
    <div className="flex gap-4">
      <div className="flex flex-col items-center">
        <div className="w-10 h-10 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center z-10 shadow-sm">
          {icon}
        </div>
        {!isLast && <div className="w-[1px] h-10 bg-gradient-to-b from-slate-700 to-transparent mt-1" />}
      </div>
      <div className="pt-2 pb-6">
        <div className="font-semibold text-slate-200">{title}</div>
        <div className="text-sm text-slate-500">{desc}</div>
      </div>
    </div>
  );
}

function AlertIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-amber-400">
      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
      <path d="M12 9v4"/>
      <path d="M12 17h.01"/>
    </svg>
  );
}
