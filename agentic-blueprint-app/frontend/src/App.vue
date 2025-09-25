<template>
  <div id="app">
    <header class="app-header">
      <h1>🤖 Agentic AI Blueprint Analyzer</h1>
      <p class="subtitle">Enter your business use case to discover the required AI components</p>
    </header>

    <main class="app-main">
      <!-- Input Section -->
      <section class="input-section">
        <div class="form-container">
          <h2>Describe Your Use Case</h2>
          <form @submit.prevent="analyzeUseCase">
            <textarea
              v-model="useCase"
              placeholder="Example: I need to build a customer service chatbot that can access our CRM system, remember past conversations, and provide accurate responses based on our internal documentation..."
              :disabled="isLoading"
              rows="5"
              minlength="10"
              maxlength="5000"
              required
            ></textarea>
            <button type="submit" :disabled="isLoading || !useCase.trim()" class="submit-btn">
              <span v-if="!isLoading">Analyze Use Case</span>
              <span v-else class="loading">
                <span class="spinner"></span>
                Analyzing...
              </span>
            </button>
          </form>
          <div v-if="error" class="error-message">
            {{ error }}
          </div>
        </div>
      </section>

      <!-- Blueprint Visualization -->
      <section class="blueprint-section">
        <h2>AI Architecture Blueprint</h2>
        <div class="blueprint-container">
          <!-- Input Layer -->
          <div class="blueprint-layer input-layer">
            <div class="component-box input-box" :class="{ active: hasAnalysis }">
              <span class="icon">📥</span>
              <span>User Input</span>
            </div>
          </div>

          <!-- Decision Layer -->
          <div class="blueprint-layer decision-layer">
            <div class="component-box complexity-box" :class="{ active: hasAnalysis }">
              <span class="icon">🔀</span>
              <span>Complexity Check</span>
            </div>
          </div>

          <!-- Processing Layer -->
          <div class="blueprint-layer processing-layer">
            <div 
              class="component-box reasoning-box"
              :class="{ active: components.reasoning_engine, highlight: components.reasoning_engine }"
            >
              <span class="icon">🧠</span>
              <span>LangGraph</span>
              <div class="label">Reasoning Engine</div>
            </div>

            <div 
              class="component-box simple-box"
              :class="{ active: components.simple_direct_answer, highlight: components.simple_direct_answer }"
            >
              <span class="icon">⚡</span>
              <span>Direct Answer</span>
              <div class="label">Simple Response</div>
            </div>
          </div>

          <!-- Agents Layer -->
          <div class="blueprint-layer agents-layer">
            <div 
              v-for="agent in ['agent_a', 'agent_b', 'agent_c']"
              :key="agent"
              class="component-box agent-box"
              :class="{ 
                active: components.agents && components.agents.includes(agent),
                highlight: components.agents && components.agents.includes(agent)
              }"
            >
              <span class="icon">🤖</span>
              <span>{{ agent.replace('_', ' ').toUpperCase() }}</span>
            </div>
          </div>

          <!-- Services Layer -->
          <div class="blueprint-layer services-layer">
            <div 
              class="component-box memory-box"
              :class="{ active: components.memory, highlight: components.memory }"
            >
              <span class="icon">💾</span>
              <span>Redis</span>
              <div class="label">Memory Management</div>
            </div>

            <div 
              class="component-box rag-box"
              :class="{ active: components.rag, highlight: components.rag }"
            >
              <span class="icon">📚</span>
              <span>RAG Pipeline</span>
              <div class="label">Internal Data</div>
            </div>

            <div 
              class="component-box mcp-box"
              :class="{ active: components.mcp_integration, highlight: components.mcp_integration }"
            >
              <span class="icon">🔌</span>
              <span>MCP Server</span>
              <div class="label">System Integration</div>
            </div>

            <div 
              class="component-box eval-box"
              :class="{ active: components.evaluation, highlight: components.evaluation }"
            >
              <span class="icon">📊</span>
              <span>RAGAS</span>
              <div class="label">Evaluation</div>
            </div>
          </div>

          <!-- Output Layer -->
          <div class="blueprint-layer output-layer">
            <div class="component-box output-box" :class="{ active: hasAnalysis }">
              <span class="icon">📤</span>
              <span>Response</span>
            </div>
          </div>

          <!-- Observability Layer (Always Active) -->
          <div class="observability-layer" :class="{ active: components.observability }">
            <div class="observability-box">
              <span class="icon">🔍</span>
              <span>Observability Layer</span>
              <div class="observability-items">
                <span>Logging</span>
                <span>•</span>
                <span>Tracing</span>
                <span>•</span>
                <span>Metrics</span>
                <span>•</span>
                <span>Monitoring</span>
              </div>
            </div>
          </div>

          <!-- Connection Lines (SVG) -->
          <svg class="connections" v-if="hasAnalysis">
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <polygon points="0 0, 10 3, 0 6" :fill="connectionColor" />
              </marker>
            </defs>
            <!-- Dynamic connection paths based on active components -->
            <path
              v-for="(path, index) in activePaths"
              :key="index"
              :d="path"
              class="connection-path"
              :class="{ animated: true }"
            />
          </svg>
        </div>

        <!-- Component Summary -->
        <div v-if="hasAnalysis" class="component-summary">
          <h3>Required Components</h3>
          <div class="summary-grid">
            <div v-for="(value, key) in getActiveComponents()" :key="key" class="summary-item">
              <span class="summary-icon">✅</span>
              <span class="summary-text">{{ formatComponentName(key) }}</span>
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>

<script>
import { ref, computed, reactive } from 'vue'
import axios from 'axios'
import config from './config'

export default {
  name: 'App',
  setup() {
    const useCase = ref('')
    const isLoading = ref(false)
    const error = ref('')
    const components = reactive({
      reasoning_engine: false,
      memory: false,
      rag: false,
      evaluation: false,
      mcp_integration: false,
      observability: true,
      simple_direct_answer: false,
      agents: []
    })

    const hasAnalysis = computed(() => {
      return Object.keys(components).some(key => {
        if (key === 'agents') return components[key].length > 0
        return components[key] === true
      })
    })

    const connectionColor = computed(() => '#10b981')

    const activePaths = computed(() => {
      const paths = []
      
      // Basic flow paths
      if (hasAnalysis.value) {
        // Input to Complexity
        paths.push('M 150 100 L 150 180')
        
        if (components.simple_direct_answer) {
          // Complexity to Simple
          paths.push('M 150 220 Q 250 250 350 280')
        }
        
        if (components.reasoning_engine) {
          // Complexity to Reasoning
          paths.push('M 150 220 Q 50 250 50 280')
        }
        
        if (components.agents.length > 0) {
          // Reasoning to Agents
          components.agents.forEach((agent, index) => {
            const xPos = 50 + (index * 150)
            paths.push(`M 50 320 Q ${xPos} 350 ${xPos} 380`)
          })
        }
        
        // Services connections
        if (components.memory) {
          paths.push('M 150 420 L 50 480')
        }
        if (components.rag) {
          paths.push('M 150 420 L 150 480')
        }
        if (components.mcp_integration) {
          paths.push('M 150 420 L 250 480')
        }
        if (components.evaluation) {
          paths.push('M 150 420 L 350 480')
        }
        
        // To output
        paths.push('M 150 520 L 150 580')
      }
      
      return paths
    })

    const analyzeUseCase = async () => {
      if (!useCase.value.trim()) return

      isLoading.value = true
      error.value = ''

      try {
        const response = await axios.post(`${config.API_BASE_URL}${config.ENDPOINTS.ANALYZE_USECASE}`, {
          usecase: useCase.value
        })

        // Update components with response
        Object.assign(components, response.data)
        
        // Ensure observability is always true
        components.observability = true

      } catch (err) {
        console.error('Error analyzing use case:', err)
        error.value = err.response?.data?.error || 'Failed to analyze use case. Please try again.'
      } finally {
        isLoading.value = false
      }
    }

    const getActiveComponents = () => {
      const active = {}
      Object.entries(components).forEach(([key, value]) => {
        if (key === 'agents' && value.length > 0) {
          active[key] = value
        } else if (value === true && key !== 'observability') {
          active[key] = value
        }
      })
      return active
    }

    const formatComponentName = (key) => {
      const names = {
        reasoning_engine: 'LangGraph Reasoning Engine',
        memory: 'Redis Memory Management',
        rag: 'RAG Pipeline',
        evaluation: 'RAGAS Evaluation',
        mcp_integration: 'MCP Server Integration',
        simple_direct_answer: 'Direct Answer',
        agents: 'Domain Agents'
      }
      return names[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    }

    return {
      useCase,
      isLoading,
      error,
      components,
      hasAnalysis,
      connectionColor,
      activePaths,
      analyzeUseCase,
      getActiveComponents,
      formatComponentName
    }
  }
}
</script>

<style scoped>
/* Component styles are in the separate style.scss file */
</style>