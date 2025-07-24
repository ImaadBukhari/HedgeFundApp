import { useNavigate } from 'react-router-dom';

export default function TrendingValueStrategy() {
  const navigate = useNavigate();

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <button style={styles.backButton} onClick={() => navigate('/dashboard')}>
          ← Back to Dashboard
        </button>
        <h1 style={styles.title}>Trending Value Strategy</h1>
      </div>

      <div style={styles.card}>
        <div style={styles.strategyHeader}>
          <div style={styles.iconContainer}>
            <div style={{...styles.icon, background: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)'}}>
              💎
            </div>
          </div>
          <div>
            <h2 style={styles.strategyTitle}>Quantitative Deep Value with Momentum</h2>
            <p style={styles.strategySubtitle}>Systematic identification of undervalued stocks with upward momentum - avoiding value traps</p>
          </div>
        </div>

        <div style={styles.content}>
          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Strategy Overview</h3>
            <p style={styles.description}>
              The Trending Value strategy identifies undervalued U.S. stocks that are also exhibiting upward price momentum, blending the best of value and growth investing. The approach begins by ranking thousands of stocks using six comprehensive value ratios: Price-to-Earnings (P/E), Price-to-Book (P/B), Price-to-Sales (P/S), Price-to-Free Cash Flow (P/FCF), Enterprise Value-to-EBITDA (EV/EBITDA), and Shareholder Yield (dividends + buybacks). From the top-ranked value stocks, the strategy then filters for those with strong recent price momentum and positive earnings revisions—helping avoid value traps and identify companies experiencing improving fundamentals.
            </p>
          </div>

          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>How It Works</h3>
            <div style={styles.featureGrid}>
              <div style={styles.feature}>
                <div style={styles.featureIcon}>📊</div>
                <h4 style={styles.featureTitle}>Six-Factor Value Ranking</h4>
                <p style={styles.featureDescription}>
                  Systematically ranks thousands of stocks using P/E, P/B, P/S, P/FCF, EV/EBITDA, and Shareholder Yield ratios
                </p>
              </div>
              <div style={styles.feature}>
                <div style={styles.featureIcon}>📈</div>
                <h4 style={styles.featureTitle}>Momentum Overlay</h4>
                <p style={styles.featureDescription}>
                  Filters for positive price momentum and earnings revisions to avoid value traps
                </p>
              </div>
              <div style={styles.feature}>
                <div style={styles.featureIcon}>⚖️</div>
                <h4 style={styles.featureTitle}>Equal Weight Portfolio</h4>
                <p style={styles.featureDescription}>
                  Constructs 25 equally weighted positions with periodic rebalancing
                </p>
              </div>
            </div>
          </div>

          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Key Metrics</h3>
            <div style={styles.metricsGrid}>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Portfolio Size</span>
                <span style={styles.metricValue}>15-25 stocks</span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Weighting Method</span>
                <span style={styles.metricValue}>Equal Weight</span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Market Focus</span>
                <span style={styles.metricValue}>U.S. Equities</span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Rebalancing</span>
                <span style={styles.metricValue}>Quarterly</span>
              </div>
            </div>
          </div>

          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Selection Criteria</h3>
            <div style={styles.criteriaGrid}>
              <div style={styles.criteriaItem}>
                <span style={styles.criteriaLabel}>Value Ratios</span>
                <span style={styles.criteriaValue}>P/E, P/B, P/S, P/FCF, EV/EBITDA rankings</span>
              </div>
              <div style={styles.criteriaItem}>
                <span style={styles.criteriaLabel}>Shareholder Yield</span>
                <span style={styles.criteriaValue}>Dividends + buybacks as % of market cap</span>
              </div>
              <div style={styles.criteriaItem}>
                <span style={styles.criteriaLabel}>Price Momentum</span>
                <span style={styles.criteriaValue}>Positive recent price trends</span>
              </div>
              <div style={styles.criteriaItem}>
                <span style={styles.criteriaLabel}>Earnings Revisions</span>
                <span style={styles.criteriaValue}>Positive analyst estimate revisions</span>
              </div>
            </div>
          </div>

          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Risk Profile</h3>
            <div style={styles.riskContainer}>
              <div style={styles.riskLevel}>
                <span style={styles.riskLabel}>Risk Level:</span>
                <div style={styles.riskBar}>
                  <div style={{...styles.riskFill, width: '55%', background: '#8b5cf6'}}></div>
                </div>
                <span style={styles.riskText}>Moderate</span>
              </div>
              <p style={styles.riskDescription}>
                Balanced approach combining value discipline with momentum validation. Lower risk than pure growth strategies due to value focus, but momentum overlay helps avoid prolonged value traps. Concentrated portfolio of 15-25 positions may create higher volatility than broad market indices.
              </p>
            </div>
          </div>

          <div style={styles.performanceBox}>
            <div style={styles.performanceIcon}>🎯</div>
            <div>
              <h4 style={styles.performanceTitle}>Strategy Advantages</h4>
              <p style={styles.performanceText}>
                This quantitative approach has historically outperformed both pure value and growth strategies by focusing on "cheap stocks that are starting to win." The momentum overlay helps identify when undervalued companies are beginning to be recognized by the market, potentially capturing both value reversion and momentum continuation.
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
    background: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)',
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