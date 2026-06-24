
import { Activity, LogOut, User as UserIcon } from 'lucide-react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

export default function Header() {
  const { user, isAuthenticated, logoutUser } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logoutUser();
    navigate('/login');
  };

  const linkStyle = ({ isActive }: { isActive: boolean }) => ({
    padding: '0.4rem 0.8rem',
    borderRadius: '0.5rem',
    fontSize: '0.875rem',
    fontWeight: 600,
    color: isActive ? '#fff' : '#94a3b8',
    background: isActive ? 'rgba(6,182,212,0.15)' : 'transparent',
    textDecoration: 'none',
    transition: 'all 0.2s',
    whiteSpace: 'nowrap' as const
  });

  return (
    <header
      style={{
        borderBottom: '1px solid var(--color-border)',
        background: 'rgba(11,15,26,0.85)',
        backdropFilter: 'blur(12px)',
        position: 'sticky', top: 0, zIndex: 50,
      }}
    >
      <div
        style={{
          maxWidth: '1400px', margin: '0 auto',
          padding: '0 1.5rem',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          height: '64px',
        }}
      >
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div
            style={{
              width: '36px', height: '36px', borderRadius: '10px',
              background: 'linear-gradient(135deg, #06b6d4, #10b981)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 0 16px rgba(6,182,212,0.4)',
            }}
          >
            <Activity size={20} color="#fff" />
          </div>
          <div>
            <div style={{ fontWeight: 800, fontSize: '1.1rem', color: '#e2e8f0', letterSpacing: '-0.02em' }}>
              DemandFlow <span className="gradient-text">AI</span>
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--color-muted)', letterSpacing: '0.08em' }}>
              DEMAND FORECASTING PLATFORM
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav style={{ display: 'flex', gap: '0.5rem', flex: 1, marginLeft: '3rem', overflowX: 'auto', scrollbarWidth: 'none', msOverflowStyle: 'none' }} className="hidden sm:flex items-center">
          <NavLink to="/" style={linkStyle}>Home</NavLink>
          {isAuthenticated ? (
            <>
              <NavLink to="/dashboard" style={linkStyle}>Dashboard</NavLink>
              <NavLink to="/history" style={linkStyle}>History</NavLink>
              <NavLink to="/scenarios" style={linkStyle}>Scenario Planner</NavLink>
            </>
          ) : (
            <>
              <NavLink to="/login" style={linkStyle}>Login</NavLink>
              <NavLink to="/register" style={linkStyle}>Register</NavLink>
            </>
          )}
        </nav>

        {/* Right area: user state + Store badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }} className="hidden md:flex">
          {isAuthenticated && user && (
            <div className="flex items-center gap-4 border-r border-white/10 pr-6">
              <div className="flex items-center gap-2 text-sm text-gray-300">
                <UserIcon className="h-4 w-4 text-emerald-400" />
                <span className="font-medium">{user.name}</span>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center gap-1.5 text-xs font-medium text-gray-400 hover:text-white transition-colors"
              >
                <LogOut className="h-3.5 w-3.5" />
                Logout
              </button>
            </div>
          )}
          
          <div
            style={{
              padding: '0.3rem 0.75rem', borderRadius: '9999px',
              background: 'rgba(6,182,212,0.12)',
              border: '1px solid rgba(6,182,212,0.25)',
              fontSize: '0.75rem', fontWeight: 600, color: '#06b6d4',
              whiteSpace: 'nowrap'
            }}
          >
            Store #1 · Rossmann
          </div>
        </div>
      </div>
      
      {/* Mobile Navigation Row */}
      <div className="sm:hidden flex flex-col gap-2 border-t border-slate-800 pb-3">
        <div className="flex overflow-x-auto gap-2 px-6 pt-3" style={{ scrollbarWidth: 'none' }}>
          <NavLink to="/" style={linkStyle}>Home</NavLink>
          {isAuthenticated ? (
            <>
              <NavLink to="/dashboard" style={linkStyle}>Dashboard</NavLink>
              <NavLink to="/history" style={linkStyle}>History</NavLink>
              <NavLink to="/scenarios" style={linkStyle}>Scenario Planner</NavLink>
            </>
          ) : (
            <>
              <NavLink to="/login" style={linkStyle}>Login</NavLink>
              <NavLink to="/register" style={linkStyle}>Register</NavLink>
            </>
          )}
        </div>
        {isAuthenticated && user && (
          <div className="flex items-center justify-between px-6 pt-2 border-t border-slate-800/50">
            <div className="flex items-center gap-2 text-sm text-gray-300">
              <UserIcon className="h-4 w-4 text-emerald-400" />
              <span className="font-medium">{user.name}</span>
            </div>
            <button
              onClick={handleLogout}
              className="flex items-center gap-1.5 text-xs font-medium text-gray-400 hover:text-white transition-colors"
            >
              <LogOut className="h-3.5 w-3.5" />
              Logout
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
