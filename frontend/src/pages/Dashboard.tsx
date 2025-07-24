import { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

export default function Dashboard() {
  const navigate = useNavigate();
  const [user, setUser] = useState<any>(null);
  const [allocations, setAllocations] = useState({
    ijr: 3000,
    leveraged_midcap: 3000,
    trending_value: 4000
  });
  const [loading, setLoading] = useState(true);
  const [isDragging, setIsDragging] = useState<string | null>(null);
  const dragRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});

  useEffect(() => {
    const userData = localStorage.getItem('user');
    if (!userData) {
      navigate('/');
      return;
    }
    
    const parsedUser = JSON.parse(userData);
    setUser(parsedUser);
    
    // Load user's allocations from database
    loadAllocations(parsedUser.id);
  }, [navigate]);

  const loadAllocations = async (userId: string) => {
    try {
      const response = await axios.get(`http://localhost:8000/api/users/${userId}/allocations`);
      if (response.data.allocations) {
        setAllocations(response.data.allocations);
      }
    } catch (error) {
      console.log('No allocations found, using defaults');
    } finally {
      setLoading(false);
    }
  };

  const updateAllocation = (key: string, value: number) => {
    setAllocations(prev => ({ ...prev, [key]: Math.max(0, value) }));
  };

  const saveAllocations = async () => {
    try {
      await axios.put(`http://localhost:8000/api/users/${user.id}/allocations`, {
        allocations
      });
      alert('Allocations saved successfully!');
    } catch (error) {
      alert('Failed to save allocations');
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('user');
    navigate('/');
  };

  const navigateToStrategy = (strategy: string) => {
    navigate(`/strategy/${strategy}`);
  };

  const navigateToManage = (strategy: string) => {
    navigate(`/manage/${strategy}`);
  };

  // Calculate total and max for dynamic scaling
  const total = allocations.ijr + allocations.leveraged_midcap + allocations.trending_value;
  const maxValue = Math.max(allocations.ijr, allocations.leveraged_midcap, allocations.trending_value);
  const scale = maxValue > 0 ? maxValue : 1;

  // Handle drag functionality
  const handleMouseDown = (key: string, e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(key);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging) return;
    
    const container = dragRefs.current[isDragging]?.parentElement;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const containerWidth = rect.width - 40; // Account for padding
    const mouseX = e.clientX - rect.left - 20; // Account for padding
    const percentage = Math.max(0, Math.min(1, mouseX / containerWidth));
    const newValue = Math.round(percentage * scale);
    
    updateAllocation(isDragging, newValue);
  };

  const handleMouseUp = () => {
    setIsDragging(null);
  };

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, scale]);

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.loading}>Loading your portfolio...</div>
      </div>
    );
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  // Placeholder chart component
  const PlaceholderChart = ({ color, data }: { color: string; data: number[] }) => (
    <div style={styles.chartContainer}>
      <svg width="100%" height="120" viewBox="0 0 300 120">
        <defs>
          <linearGradient id={`gradient-${color}`} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor={color} stopOpacity="0.3"/>
            <stop offset="100%" stopColor={color} stopOpacity="0.1"/>
          </linearGradient>
        </defs>
        {/* Chart line */}
        <polyline
          fill={`url(#gradient-${color})`}
          stroke={color}
          strokeWidth="2"
          points={data.map((value, index) => 
            `${(index / (data.length - 1)) * 300},${120 - (value * 80)}`
          ).join(' ')}
        />
        <polyline
          fill="none"
          stroke={color}
          strokeWidth="2"
          points={data.map((value, index) => 
            `${(index / (data.length - 1)) * 300},${120 - (value * 80)}`
          ).join(' ')}
        />
      </svg>
    </div>
  );

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>Portfolio Dashboard</h1>
        <div style={styles.headerRight}>
          <span style={styles.welcome}>Welcome, {user?.email}</span>
          <button style={styles.logoutBtn} onClick={handleLogout}>
            Logout
          </button>
        </div>
      </div>

      {/* Compact Asset Allocation Section */}
      <div style={styles.compactCard}>
        <h2 style={styles.compactCardTitle}>Asset Allocation</h2>
        <div style={styles.compactAllocationsContainer}>
          {/* IJR Bar */}
          <div style={styles.compactAllocationRow}>
            <div style={styles.compactLabelContainer}>
              <span style={styles.compactStrategyLabel}>IJR</span>
              <input
                type="number"
                value={allocations.ijr}
                onChange={(e) => updateAllocation('ijr', parseInt(e.target.value) || 0)}
                style={styles.compactDollarInput}
                min="0"
                step="100"
              />
            </div>
            <div style={styles.compactBarContainer} ref={el => dragRefs.current['ijr'] = el}>
              <div 
                style={{
                  ...styles.compactBar,
                  width: `${(allocations.ijr / scale) * 100}%`,
                  background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)'
                }}
              />
              <div
                style={{
                  ...styles.compactDragHandle,
                  left: `${(allocations.ijr / scale) * 100}%`,
                  background: 'rgba(16, 185, 129, 0.8)',
                }}
                onMouseDown={(e) => handleMouseDown('ijr', e)}
              />
            </div>
            <div style={styles.compactValueDisplay}>{formatCurrency(allocations.ijr)}</div>
          </div>

          {/* Leveraged Midcap Bar */}
          <div style={styles.compactAllocationRow}>
            <div style={styles.compactLabelContainer}>
              <span style={styles.compactStrategyLabel}>Leveraged Midcap</span>
              <input
                type="number"
                value={allocations.leveraged_midcap}
                onChange={(e) => updateAllocation('leveraged_midcap', parseInt(e.target.value) || 0)}
                style={styles.compactDollarInput}
                min="0"
                step="100"
              />
            </div>
            <div style={styles.compactBarContainer} ref={el => dragRefs.current['leveraged_midcap'] = el}>
              <div 
                style={{
                  ...styles.compactBar,
                  width: `${(allocations.leveraged_midcap / scale) * 100}%`,
                  background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'
                }}
              />
              <div
                style={{
                  ...styles.compactDragHandle,
                  left: `${(allocations.leveraged_midcap / scale) * 100}%`,
                  background: 'rgba(245, 158, 11, 0.8)',
                }}
                onMouseDown={(e) => handleMouseDown('leveraged_midcap', e)}
              />
            </div>
            <div style={styles.compactValueDisplay}>{formatCurrency(allocations.leveraged_midcap)}</div>
          </div>

          {/* Trending Value Bar */}
          <div style={styles.compactAllocationRow}>
            <div style={styles.compactLabelContainer}>
              <span style={styles.compactStrategyLabel}>Trending Value</span>
              <input
                type="number"
                value={allocations.trending_value}
                onChange={(e) => updateAllocation('trending_value', parseInt(e.target.value) || 0)}
                style={styles.compactDollarInput}
                min="0"
                step="100"
              />
            </div>
            <div style={styles.compactBarContainer} ref={el => dragRefs.current['trending_value'] = el}>
              <div 
                style={{
                  ...styles.compactBar,
                  width: `${(allocations.trending_value / scale) * 100}%`,
                  background: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)'
                }}
              />
              <div
                style={{
                  ...styles.compactDragHandle,
                  left: `${(allocations.trending_value / scale) * 100}%`,
                  background: 'rgba(139, 92, 246, 0.8)',
                }}
                onMouseDown={(e) => handleMouseDown('trending_value', e)}
              />
            </div>
            <div style={styles.compactValueDisplay}>{formatCurrency(allocations.trending_value)}</div>
          </div>
        </div>

        <div style={styles.totalContainer}>
          <span style={styles.totalLabel}>Total: {formatCurrency(total)}</span>
          <button style={styles.saveButton} onClick={saveAllocations}>
            Save
          </button>
        </div>
      </div>

      {/* Strategy Management Cards */}
      <div style={styles.strategiesGrid}>
        {/* IJR Strategy Card */}
        <div style={styles.strategyCard}>
          <div style={styles.strategyHeader}>
            <div style={styles.strategyIcon}>📈</div>
            <div>
              <h3 style={styles.strategyTitle}>IJR Value Averaging</h3>
              <p style={styles.strategySubtitle}>Small-cap systematic investing</p>
            </div>
          </div>
          
          <PlaceholderChart 
            color="#10b981" 
            data={[0.2, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7, 0.9, 0.85, 1.0]}
          />
          
          <div style={styles.strategyStats}>
            <div style={styles.stat}>
              <span style={styles.statLabel}>Allocation</span>
              <span style={styles.statValue}>{formatCurrency(allocations.ijr)}</span>
            </div>
            <div style={styles.stat}>
              <span style={styles.statLabel}>Return (YTD)</span>
              <span style={{...styles.statValue, color: '#10b981'}}>+12.4%</span>
            </div>
          </div>
          
          <div style={styles.strategyActions}>
            <button 
              style={{...styles.actionButton, ...styles.primaryButton}}
              onClick={() => navigateToManage('ijr')}
            >
              Manage Strategy
            </button>
            <button 
              style={{...styles.actionButton, ...styles.secondaryButton}}
              onClick={() => navigateToStrategy('ijr')}
            >
              Learn More
            </button>
          </div>
        </div>

        {/* Leveraged Midcap Strategy Card */}
        <div style={styles.strategyCard}>
          <div style={styles.strategyHeader}>
            <div style={styles.strategyIcon}>⚡</div>
            <div>
              <h3 style={styles.strategyTitle}>Leveraged Midcap</h3>
              <p style={styles.strategySubtitle}>2x amplified growth exposure</p>
            </div>
          </div>
          
          <PlaceholderChart 
            color="#f59e0b" 
            data={[0.1, 0.3, 0.2, 0.7, 0.4, 0.9, 0.6, 0.8, 0.95, 0.7]}
          />
          
          <div style={styles.strategyStats}>
            <div style={styles.stat}>
              <span style={styles.statLabel}>Allocation</span>
              <span style={styles.statValue}>{formatCurrency(allocations.leveraged_midcap)}</span>
            </div>
            <div style={styles.stat}>
              <span style={styles.statLabel}>Return (YTD)</span>
              <span style={{...styles.statValue, color: '#10b981'}}>+24.7%</span>
            </div>
          </div>
          
          <div style={styles.strategyActions}>
            <button 
              style={{...styles.actionButton, ...styles.primaryButton}}
              onClick={() => navigateToManage('leveraged-midcap')}
            >
              Manage Strategy
            </button>
            <button 
              style={{...styles.actionButton, ...styles.secondaryButton}}
              onClick={() => navigateToStrategy('leveraged-midcap')}
            >
              Learn More
            </button>
          </div>
        </div>

        {/* Trending Value Strategy Card */}
        <div style={styles.strategyCard}>
          <div style={styles.strategyHeader}>
            <div style={styles.strategyIcon}>💎</div>
            <div>
              <h3 style={styles.strategyTitle}>Trending Value</h3>
              <p style={styles.strategySubtitle}>AI-powered value discovery</p>
            </div>
          </div>
          
          <PlaceholderChart 
            color="#8b5cf6" 
            data={[0.3, 0.2, 0.5, 0.4, 0.7, 0.6, 0.8, 0.9, 0.75, 0.85]}
          />
          
          <div style={styles.strategyStats}>
            <div style={styles.stat}>
              <span style={styles.statLabel}>Allocation</span>
              <span style={styles.statValue}>{formatCurrency(allocations.trending_value)}</span>
            </div>
            <div style={styles.stat}>
              <span style={styles.statLabel}>Return (YTD)</span>
              <span style={{...styles.statValue, color: '#10b981'}}>+18.2%</span>
            </div>
          </div>
          
          <div style={styles.strategyActions}>
            <button 
              style={{...styles.actionButton, ...styles.primaryButton}}
              onClick={() => navigateToManage('trending-value')}
            >
              Manage Strategy
            </button>
            <button 
              style={{...styles.actionButton, ...styles.secondaryButton}}
              onClick={() => navigateToStrategy('trending-value')}
            >
              Learn More
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    padding: '2rem',
  },
  loading: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100vh',
    color: 'white',
    fontSize: '1.2rem',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '2rem',
  },
  title: {
    color: 'white',
    fontSize: '2.5rem',
    fontWeight: 'bold',
    margin: 0,
  },
  headerRight: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
  },
  welcome: {
    color: 'white',
    fontSize: '1rem',
  },
  logoutBtn: {
    background: 'rgba(255, 255, 255, 0.2)',
    color: 'white',
    border: '1px solid rgba(255, 255, 255, 0.3)',
    padding: '0.5rem 1rem',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '0.875rem',
    transition: 'background 0.2s',
  },
  // Compact allocation styles
  compactCard: {
    background: 'white',
    borderRadius: '12px',
    padding: '1.5rem',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
    maxWidth: '800px',
    margin: '0 auto 2rem auto',
  },
  compactCardTitle: {
    fontSize: '1.5rem',
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: '1rem',
  },
  compactAllocationsContainer: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '1rem',
    marginBottom: '1rem',
  },
  compactAllocationRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
  },
  compactLabelContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    minWidth: '200px',
  },
  compactStrategyLabel: {
    color: '#374151',
    fontSize: '0.875rem',
    fontWeight: '600',
    minWidth: '120px',
  },
  compactDollarInput: {
    width: '80px',
    padding: '0.25rem 0.5rem',
    border: '1px solid #d1d5db',
    borderRadius: '4px',
    fontSize: '0.75rem',
  },
  compactBarContainer: {
    height: '20px',
    background: '#f3f4f6',
    borderRadius: '10px',
    overflow: 'visible' as const,
    position: 'relative' as const,
    flex: 1,
    cursor: 'pointer',
  },
  compactBar: {
    height: '100%',
    borderRadius: '10px',
    transition: 'width 0.3s ease',
    position: 'relative' as const,
  },
  compactDragHandle: {
    position: 'absolute' as const,
    top: '50%',
    transform: 'translateY(-50%) translateX(-50%)',
    width: '16px',
    height: '16px',
    borderRadius: '50%',
    cursor: 'grab',
    border: '2px solid white',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
    transition: 'left 0.3s ease',
    zIndex: 10,
  },
  compactValueDisplay: {
    fontSize: '0.75rem',
    color: '#6b7280',
    fontWeight: '600',
    minWidth: '80px',
    textAlign: 'right' as const,
  },
  totalContainer: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '0.75rem',
    background: '#f9fafb',
    borderRadius: '6px',
  },
  totalLabel: {
    fontSize: '1rem',
    fontWeight: '600',
    color: '#374151',
  },
  saveButton: {
    padding: '0.5rem 1rem',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    fontSize: '0.875rem',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'transform 0.2s',
  },
  // Strategy cards
  strategiesGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
    gap: '1.5rem',
    maxWidth: '1200px',
    margin: '0 auto',
  },
  strategyCard: {
    background: 'white',
    borderRadius: '12px',
    padding: '1.5rem',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
    transition: 'transform 0.2s, box-shadow 0.2s',
    cursor: 'pointer',
  },
  strategyHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    marginBottom: '1rem',
  },
  strategyIcon: {
    fontSize: '2rem',
    width: '60px',
    height: '60px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: '12px',
    background: '#f9fafb',
  },
  strategyTitle: {
    fontSize: '1.25rem',
    fontWeight: 'bold',
    color: '#1f2937',
    margin: '0 0 0.25rem 0',
  },
  strategySubtitle: {
    fontSize: '0.875rem',
    color: '#6b7280',
    margin: 0,
  },
  chartContainer: {
    marginBottom: '1rem',
    height: '120px',
    background: '#f9fafb',
    borderRadius: '8px',
    padding: '1rem',
  },
  strategyStats: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '1rem',
    padding: '0.75rem',
    background: '#f9fafb',
    borderRadius: '6px',
  },
  stat: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
  },
  statLabel: {
    fontSize: '0.75rem',
    color: '#6b7280',
    marginBottom: '0.25rem',
  },
  statValue: {
    fontSize: '0.875rem',
    fontWeight: 'bold',
    color: '#1f2937',
  },
  strategyActions: {
    display: 'flex',
    gap: '0.5rem',
  },
  actionButton: {
    flex: 1,
    padding: '0.5rem 1rem',
    border: 'none',
    borderRadius: '6px',
    fontSize: '0.875rem',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'background 0.2s',
  },
  primaryButton: {
    background: '#3b82f6',
    color: 'white',
  },
  secondaryButton: {
    background: '#f3f4f6',
    color: '#374151',
  },
};