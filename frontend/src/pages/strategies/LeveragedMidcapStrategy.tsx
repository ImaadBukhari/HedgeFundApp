import { useNavigate } from 'react-router-dom';

export default function LeveragedMidcapStrategy() {
  const navigate = useNavigate();

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <button style={styles.backButton} onClick={() => navigate('/dashboard')}>
          ← Back to Dashboard
        </button>
        <h1 style={styles.title}>Leveraged Midcap Strategy</h1>
      </div>

      <div style={styles.card}>
        <div style={styles.strategyHeader}>
          <div style={styles.iconContainer}>
            <div style={{...styles.icon, background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'}}>
              ⚡
            </div>
          </div>
          <div>
            <h2 style={styles.strategyTitle}>Amplified Core Momentum with Risk Controls</h2>
            <p style={styles.strategySubtitle}>2x leveraged ETF exposure with technical safeguards for disciplined risk management</p>
          </div>
        </div>

        <div style={styles.content}>
          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Strategy Overview</h3>
            <p style={styles.description}>
              The Leveraged Midcap strategy seeks to generate outsized returns by investing in a 2x leveraged ETF tracking the S&P MidCap 400, such as ProShares Ultra MidCap 400 (MVV). This segment captures companies that are large enough to be operationally stable yet small enough to have significant upside potential. The use of leverage enhances exposure to positive price movements, making this strategy especially potent during strong market uptrends. However, leveraged ETFs reset daily and can quickly compound losses in volatile conditions—so this strategy incorporates three technical indicators for disciplined risk management.
            </p>
          </div>

          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>How It Works</h3>
            <div style={styles.featureGrid}>
              <div style={styles.feature}>
                <div style={styles.featureIcon}>📊</div>
                <h4 style={styles.featureTitle}>Simple Moving Average (SMA)</h4>
                <p style={styles.featureDescription}>
                  Avoids positions during extended downtrends by requiring price action above the 50-day or 200-day average
                </p>
              </div>
              <div style={styles.feature}>
                <div style={styles.featureIcon}>📈</div>
                <h4 style={styles.featureTitle}>MACD Indicator</h4>
                <p style={styles.featureDescription}>
                  Identifies bullish momentum shifts and trend reversals for optimal entry and exit timing
                </p>
              </div>
              <div style={styles.feature}>
                <div style={styles.featureIcon}>⚖️</div>
                <h4 style={styles.featureTitle}>RSI Risk Management</h4>
                <p style={styles.featureDescription}>
                  Prevents overexposure during overbought conditions and flags better entry points
                </p>
              </div>
            </div>
          </div>

          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Key Metrics</h3>
            <div style={styles.metricsGrid}>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Leverage Ratio</span>
                <span style={styles.metricValue}>2.0x</span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Target ETF</span>
                <span style={styles.metricValue}>MVV (S&P 400)</span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Risk Controls</span>
                <span style={styles.metricValue}>3 Technical Indicators</span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Daily Reset</span>
                <span style={styles.metricValue}>ETF Characteristic</span>
              </div>
            </div>
          </div>

          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Technical Safeguards</h3>
            <div style={styles.criteriaGrid}>
              <div style={styles.criteriaItem}>
                <span style={styles.criteriaLabel}>SMA Trend Filter</span>
                <span style={styles.criteriaValue}>50-day and 200-day moving averages</span>
              </div>
              <div style={styles.criteriaItem}>
                <span style={styles.criteriaLabel}>MACD Momentum</span>
                <span style={styles.criteriaValue}>Bullish crossovers and trend reversals</span>
              </div>
              <div style={styles.criteriaItem}>
                <span style={styles.criteriaLabel}>RSI Overbought/Oversold</span>
                <span style={styles.criteriaValue}>Entry/exit signals and risk mitigation</span>
              </div>
              <div style={styles.criteriaItem}>
                <span style={styles.criteriaLabel}>Market Segment</span>
                <span style={styles.criteriaValue}>S&P MidCap 400 companies</span>
              </div>
            </div>
          </div>

          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Risk Profile</h3>
            <div style={styles.riskContainer}>
              <div style={styles.riskLevel}>
                <span style={styles.riskLabel}>Risk Level:</span>
                <div style={styles.riskBar}>
                  <div style={{...styles.riskFill, width: '65%', background: '#ffa500'}}></div>
                </div>
                <span style={styles.riskText}>High</span>
              </div>
              <p style={styles.riskDescription}>
                High-conviction but rules-based strategy. Leveraged ETFs amplify both gains and losses due to daily rebalancing. 
                Technical safeguards help manage risk, but volatility can still compound quickly in adverse conditions. 
                Especially potent during strong market uptrends.
              </p>
            </div>
          </div>

          <div style={styles.warningBox}>
            <div style={styles.warningIcon}>⚠️</div>
            <div>
              <h4 style={styles.warningTitle}>Leveraged ETF Risk Disclosure</h4>
              <p style={styles.warningText}>
                Leveraged ETFs reset daily and can quickly compound losses in volatile conditions. This strategy combines 
                leveraged exposure with dynamic safeguards to enhance returns without taking on uncontrolled risk. 
                Suitable for experienced investors during favorable market conditions.
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
    background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
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
    minWidth: '80px',
  },
  riskDescription: {
    color: '#6b7280',
    fontSize: '0.875rem',
    lineHeight: '1.5',
    margin: 0,
  },
  warningBox: {
    display: 'flex',
    gap: '1rem',
    padding: '1.5rem',
    background: '#fef3cd',
    border: '1px solid #f59e0b',
    borderRadius: '8px',
  },
  warningIcon: {
    fontSize: '1.5rem',
    flexShrink: 0,
  },
  warningTitle: {
    color: '#92400e',
    fontSize: '1.125rem',
    fontWeight: '600',
    margin: '0 0 0.5rem 0',
  },
  warningText: {
    color: '#92400e',
    fontSize: '0.875rem',
    lineHeight: '1.5',
    margin: 0,
  },
};