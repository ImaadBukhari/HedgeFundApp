import { useNavigate } from 'react-router-dom';

export default function IJRStrategy() {
  const navigate = useNavigate();

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <button style={styles.backButton} onClick={() => navigate('/dashboard')}>
          ← Back to Dashboard
        </button>
        <h1 style={styles.title}>IJR Strategy</h1>
      </div>

      <div style={styles.card}>
        <div style={styles.strategyHeader}>
          <div style={styles.iconContainer}>
            <div style={{...styles.icon, background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)'}}>
              📈
            </div>
          </div>
          <div>
            <h2 style={styles.strategyTitle}>IJR Value Averaging</h2>
            <p style={styles.strategySubtitle}>Systematic wealth building through strategic buy pressure on small-cap equities</p>
          </div>
        </div>

        <div style={styles.content}>
          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Strategy Overview</h3>
            <p style={styles.description}>
              The IJR Value Averaging strategy involves investing in the iShares Core S&P Small-Cap ETF (IJR) using a 
              sophisticated value averaging approach rather than traditional dollar-cost averaging. Instead of contributing 
              a fixed amount at regular intervals, value averaging targets a steadily increasing portfolio value over time. 
              Contributions vary—buying more shares when prices fall and less when prices rise—thereby automatically 
              enforcing a "buy low, buy less when high" discipline that takes advantage of market volatility.
            </p>
          </div>

          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>How It Works</h3>
            <div style={styles.featureGrid}>
              <div style={styles.feature}>
                <div style={styles.featureIcon}>📊</div>
                <h4 style={styles.featureTitle}>Value Averaging</h4>
                <p style={styles.featureDescription}>
                  Targets steadily increasing portfolio value, automatically buying more during downturns and less during surges
                </p>
              </div>
              <div style={styles.feature}>
                <div style={styles.featureIcon}>🎯</div>
                <h4 style={styles.featureTitle}>Contrarian Pressure</h4>
                <p style={styles.featureDescription}>
                  Invests more aggressively during market downturns when valuations are attractive
                </p>
              </div>
              <div style={styles.feature}>
                <div style={styles.featureIcon}>💰</div>
                <h4 style={styles.featureTitle}>Lower Average Cost</h4>
                <p style={styles.featureDescription}>
                  Results in lower average cost per share compared to fixed contribution methods
                </p>
              </div>
            </div>
          </div>

          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Key Metrics</h3>
            <div style={styles.metricsGrid}>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Target ETF</span>
                <span style={styles.metricValue}>IJR</span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Expense Ratio</span>
                <span style={styles.metricValue}>0.06%</span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Rebalancing</span>
                <span style={styles.metricValue}>Monthly</span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Holdings</span>
                <span style={styles.metricValue}>~600 Small-Caps</span>
              </div>
            </div>
          </div>

          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Strategy Advantages</h3>
            <div style={styles.criteriaGrid}>
              <div style={styles.criteriaItem}>
                <span style={styles.criteriaLabel}>Market Inefficiency Exploitation</span>
                <span style={styles.criteriaValue}>Takes advantage of small-cap volatility for better entry points</span>
              </div>
              <div style={styles.criteriaItem}>
                <span style={styles.criteriaLabel}>Emotional Discipline</span>
                <span style={styles.criteriaValue}>Removes emotional decision-making through systematic approach</span>
              </div>
              <div style={styles.criteriaItem}>
                <span style={styles.criteriaLabel}>Enhanced Compounding</span>
                <span style={styles.criteriaValue}>Lower average costs lead to improved long-term returns</span>
              </div>
              <div style={styles.criteriaItem}>
                <span style={styles.criteriaLabel}>Diversification</span>
                <span style={styles.criteriaValue}>Broad exposure to S&P SmallCap 600 Index across sectors</span>
              </div>
            </div>
          </div>

          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Value Averaging vs Dollar-Cost Averaging</h3>
            <div style={styles.comparisonContainer}>
              <div style={styles.comparisonItem}>
                <h4 style={styles.comparisonTitle}>Traditional DCA</h4>
                <p style={styles.comparisonDescription}>
                  Fixed contributions regardless of market conditions. Simple but doesn't capitalize on volatility.
                </p>
              </div>
              <div style={styles.comparisonItem}>
                <h4 style={styles.comparisonTitle}>Value Averaging</h4>
                <p style={styles.comparisonDescription}>
                  Variable contributions based on target portfolio value. More aggressive during downturns, conservative during peaks.
                </p>
              </div>
            </div>
          </div>

          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Risk Profile</h3>
            <div style={styles.riskContainer}>
              <div style={styles.riskLevel}>
                <span style={styles.riskLabel}>Risk Level:</span>
                <div style={styles.riskBar}>
                  <div style={{...styles.riskFill, width: '35%', background: '#357a38'}}></div>
                </div>
                <span style={styles.riskText}>Moderate-High</span>
              </div>
              <p style={styles.riskDescription}>
                Small-cap stocks are inherently volatile but offer high growth potential. Value averaging helps manage 
                this volatility by enforcing disciplined buying patterns. Suitable for investors with longer time horizons 
                who can handle variable contribution amounts and market fluctuations.
              </p>
            </div>
          </div>

          <div style={styles.performanceBox}>
            <div style={styles.performanceIcon}>📈</div>
            <div>
              <h4 style={styles.performanceTitle}>Expected Advantages</h4>
              <p style={styles.performanceText}>
                Value averaging typically outperforms dollar-cost averaging by 1-2% annually due to better timing of purchases. 
                Combined with small-cap equity premiums, this strategy targets superior long-term wealth accumulation.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    padding: '2rem',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    marginBottom: '2rem',
  },
  backButton: {
    background: 'rgba(255, 255, 255, 0.2)',
    color: 'white',
    border: '1px solid rgba(255, 255, 255, 0.3)',
    padding: '0.5rem 1rem',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '0.875rem',
    transition: 'background 0.2s',
  },
  title: {
    color: 'white',
    fontSize: '2.5rem',
    fontWeight: 'bold',
    margin: 0,
  },
  card: {
    background: 'white',
    borderRadius: '12px',
    padding: '2rem',
    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    maxWidth: '900px',
    margin: '0 auto',
  },
  strategyHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '1.5rem',
    marginBottom: '3rem',
    paddingBottom: '2rem',
    borderBottom: '2px solid #f3f4f6',
  },
  iconContainer: {
    flexShrink: 0,
  },
  icon: {
    width: '80px',
    height: '80px',
    borderRadius: '20px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '2rem',
    color: 'white',
  },
  strategyTitle: {
    fontSize: '2rem',
    fontWeight: 'bold',
    color: '#1f2937',
    margin: '0 0 0.5rem 0',
  },
  strategySubtitle: {
    color: '#6b7280',
    fontSize: '1.125rem',
    margin: 0,
  },
  content: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '3rem',
  },
  section: {},
  sectionTitle: {
    fontSize: '1.5rem',
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: '1rem',
  },
  description: {
    color: '#4b5563',
    lineHeight: '1.6',
    fontSize: '1rem',
  },
  featureGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '1.5rem',
  },
  feature: {
    padding: '1.5rem',
    background: '#f9fafb',
    borderRadius: '8px',
    textAlign: 'center' as const,
  },
  featureIcon: {
    fontSize: '2rem',
    marginBottom: '1rem',
  },
  featureTitle: {
    fontSize: '1.125rem',
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: '0.5rem',
  },
  featureDescription: {
    color: '#6b7280',
    fontSize: '0.875rem',
    lineHeight: '1.5',
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '1rem',
  },
  metric: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '1rem',
    background: '#f9fafb',
    borderRadius: '6px',
  },
  metricLabel: {
    color: '#6b7280',
    fontSize: '0.875rem',
    fontWeight: '500',
  },
  metricValue: {
    color: '#1f2937',
    fontSize: '1rem',
    fontWeight: 'bold',
  },
  criteriaGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '1rem',
  },
  criteriaItem: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '0.5rem',
    padding: '1rem',
    background: '#f9fafb',
    borderRadius: '6px',
  },
  criteriaLabel: {
    color: '#374151',
    fontSize: '0.875rem',
    fontWeight: '600',
  },
  criteriaValue: {
    color: '#6b7280',
    fontSize: '0.875rem',
  },
  comparisonContainer: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '1.5rem',
  },
  comparisonItem: {
    padding: '1.5rem',
    background: '#f9fafb',
    borderRadius: '8px',
    border: '2px solid #e5e7eb',
  },
  comparisonTitle: {
    color: '#1f2937',
    fontSize: '1.125rem',
    fontWeight: '600',
    marginBottom: '0.5rem',
  },
  comparisonDescription: {
    color: '#6b7280',
    fontSize: '0.875rem',
    lineHeight: '1.5',
    margin: 0,
  },
  riskContainer: {
    padding: '1.5rem',
    background: '#f9fafb',
    borderRadius: '8px',
  },
  riskLevel: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    marginBottom: '1rem',
  },
  riskLabel: {
    color: '#374151',
    fontWeight: '600',
    minWidth: '100px',
  },
  riskBar: {
    flex: 1,
    height: '8px',
    background: '#e5e7eb',
    borderRadius: '4px',
    overflow: 'hidden',
  },
  riskFill: {
    height: '100%',
    borderRadius: '4px',
    transition: 'width 0.3s ease',
  },
  riskText: {
    color: '#374151',
    fontWeight: '600',
    minWidth: '120px',
  },
  riskDescription: {
    color: '#6b7280',
    fontSize: '0.875rem',
    lineHeight: '1.5',
    margin: 0,
  },
  performanceBox: {
    display: 'flex',
    gap: '1rem',
    padding: '1.5rem',
    background: '#f0f9ff',
    border: '1px solid #0ea5e9',
    borderRadius: '8px',
  },
  performanceIcon: {
    fontSize: '1.5rem',
    flexShrink: 0,
  },
  performanceTitle: {
    color: '#0c4a6e',
    fontSize: '1.125rem',
    fontWeight: '600',
    margin: '0 0 0.5rem 0',
  },
  performanceText: {
    color: '#0c4a6e',
    fontSize: '0.875rem',
    lineHeight: '1.5',
    margin: 0,
  },
};