// API Configuration
const config = {
  API_BASE_URL: import.meta.env.VITE_API_URL || '/api',
  ENDPOINTS: {
    ANALYZE_USECASE: '/analyze-usecase',
    HEALTH: '/health'
  }
}

export default config