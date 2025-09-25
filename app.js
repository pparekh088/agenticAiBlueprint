// Architecture Components Configuration with Professional Color Palette
const components = {
    input: {
        id: 'input',
        x: 50,
        y: 300,
        width: 120,
        height: 60,
        label: 'User Input',
        color: '#2563eb', // Bright blue for entry point
        description: 'Entry point for all user requests. Accepts queries and initiates the processing flow.'
    },
    complexity: {
        id: 'complexity',
        x: 220,
        y: 300,
        width: 140,
        height: 60,
        label: 'Complexity Check',
        color: '#8b5cf6', // Purple for decision point
        description: 'Analyzes request complexity to determine optimal processing path (simple vs complex).'
    },
    langgraph: {
        id: 'langgraph',
        x: 420,
        y: 180,
        width: 160,
        height: 60,
        label: 'LangGraph',
        color: '#ec4899', // Pink for reasoning engine
        description: 'Reasoning engine for complex decision-making. Orchestrates multi-agent workflows and decision trees.'
    },
    agentA: {
        id: 'agentA',
        x: 650,
        y: 300,
        width: 120,
        height: 60,
        label: 'Agent A',
        color: '#10b981', // Emerald for agents
        description: 'Primary processing agent. Handles simple queries and can invoke MCP servers.'
    },
    agentB: {
        id: 'agentB',
        x: 650,
        y: 180,
        width: 120,
        height: 60,
        label: 'Agent B',
        color: '#10b981', // Emerald for agents
        description: 'Specialized agent for complex analytical tasks and data processing.'
    },
    agentC: {
        id: 'agentC',
        x: 650,
        y: 100,
        width: 120,
        height: 60,
        label: 'Agent C',
        color: '#10b981', // Emerald for agents
        description: 'Advanced agent for multi-step reasoning and orchestration tasks.'
    },
    mcp: {
        id: 'mcp',
        x: 830,
        y: 300,
        width: 140,
        height: 60,
        label: 'MCP Server',
        color: '#f59e0b', // Amber for integration layer
        description: 'Model Context Protocol server for connecting to existing systems.'
    },
    systemA: {
        id: 'systemA',
        x: 1020,
        y: 380,
        width: 140,
        height: 60,
        label: 'System A',
        color: '#64748b', // Slate for external systems
        description: 'Existing system accessed through MCP server integration.'
    },
    rag: {
        id: 'rag',
        x: 420,
        y: 420,
        width: 160,
        height: 60,
        label: 'RAG on Internal Data',
        color: '#06b6d4', // Cyan for RAG
        description: 'Retrieval Augmented Generation directly on internal data sources.'
    },
    internalData: {
        id: 'internalData',
        x: 250,
        y: 420,
        width: 120,
        height: 60,
        label: 'Internal Data',
        color: '#0ea5e9', // Sky blue for data sources
        description: 'Internal knowledge base and document repositories for RAG operations.'
    },
    redis: {
        id: 'redis',
        x: 830,
        y: 180,
        width: 120,
        height: 60,
        label: 'Redis',
        color: '#ef4444', // Red for Redis
        description: 'Memory management for session state and caching agent contexts.'
    },
    ragas: {
        id: 'ragas',
        x: 650,
        y: 420,
        width: 120,
        height: 60,
        label: 'RAGAS',
        color: '#a855f7', // Purple for evaluation
        description: 'Response evaluation framework for quality assessment and accuracy scoring.'
    },
    output: {
        id: 'output',
        x: 1020,
        y: 240,
        width: 120,
        height: 60,
        label: 'Response',
        color: '#22c55e', // Green for output
        description: 'Final processed response delivered to the user with quality metrics.'
    },
    azure: {
        id: 'azure',
        x: 530,
        y: 520,
        width: 140,
        height: 50,
        label: 'Azure',
        color: '#0078d4', // Azure blue
        description: 'Cloud hosting infrastructure with auto-scaling and high availability.'
    },
    observability: {
        id: 'observability',
        x: 350,
        y: 50,
        width: 500,
        height: 40,
        label: '🔍 Observability Layer (Always Active)',
        color: '#f97316', // Orange for observability
        description: 'Comprehensive monitoring, logging, tracing, and metrics collection for all components.'
    }
};

// Connection paths configuration
const connections = {
    simple: [
        { from: 'input', to: 'complexity', id: 'path-input-complexity' },
        { from: 'complexity', to: 'agentA', id: 'path-complexity-agentA-simple' },
        { from: 'agentA', to: 'output', id: 'path-agentA-output' }
    ],
    complex: [
        { from: 'input', to: 'complexity', id: 'path-input-complexity' },
        { from: 'complexity', to: 'langgraph', id: 'path-complexity-langgraph' },
        { from: 'langgraph', to: 'agentA', id: 'path-langgraph-agentA' },
        { from: 'langgraph', to: 'agentB', id: 'path-langgraph-agentB' },
        { from: 'langgraph', to: 'agentC', id: 'path-langgraph-agentC' },
        { from: 'agentA', to: 'mcp', id: 'path-agentA-mcp' },
        { from: 'mcp', to: 'systemA', id: 'path-mcp-systemA' },
        { from: 'agentB', to: 'redis', id: 'path-agentB-redis' },
        { from: 'agentC', to: 'redis', id: 'path-agentC-redis' },
        { from: 'mcp', to: 'output', id: 'path-mcp-output' },
        { from: 'redis', to: 'output', id: 'path-redis-output' }
    ],
    rag: [
        { from: 'input', to: 'complexity', id: 'path-input-complexity' },
        { from: 'complexity', to: 'rag', id: 'path-complexity-rag' },
        { from: 'internalData', to: 'rag', id: 'path-internalData-rag' },
        { from: 'rag', to: 'ragas', id: 'path-rag-ragas' },
        { from: 'ragas', to: 'output', id: 'path-ragas-output' }
    ],
    observability: [
        { from: 'observability', to: 'complexity', id: 'obs-complexity', dashed: true },
        { from: 'observability', to: 'langgraph', id: 'obs-langgraph', dashed: true },
        { from: 'observability', to: 'agentA', id: 'obs-agentA', dashed: true },
        { from: 'observability', to: 'rag', id: 'obs-rag', dashed: true },
        { from: 'observability', to: 'redis', id: 'obs-redis', dashed: true },
        { from: 'observability', to: 'ragas', id: 'obs-ragas', dashed: true },
        { from: 'observability', to: 'mcp', id: 'obs-mcp', dashed: true }
    ]
};

// Application State
let currentUseCase = null;
let requestCount = 0;
let avgLatency = 0;
let accuracyScore = 0;
let memoryUsage = 0;

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initializeArchitecture();
    setupEventListeners();
    startMetricsSimulation();
    addLog('info', 'Architecture visualization loaded successfully');
});

// Initialize SVG architecture
function initializeArchitecture() {
    const svg = document.getElementById('architecture-svg');
    
    // Draw all connections first (behind components)
    drawAllConnections(svg);
    
    // Draw all components
    Object.values(components).forEach(comp => {
        drawComponent(svg, comp);
    });
    
    // Highlight observability by default
    highlightObservability();
}

// Draw component box
function drawComponent(svg, comp) {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.classList.add('component-box');
    g.setAttribute('data-id', comp.id);
    
    // Component rectangle
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', comp.x);
    rect.setAttribute('y', comp.y);
    rect.setAttribute('width', comp.width);
    rect.setAttribute('height', comp.height);
    rect.setAttribute('rx', '8');
    rect.setAttribute('fill', comp.color);
    rect.setAttribute('fill-opacity', '0.9');
    rect.setAttribute('stroke', comp.color);
    rect.setAttribute('stroke-width', '2');
    
    // Component text
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', comp.x + comp.width / 2);
    text.setAttribute('y', comp.y + comp.height / 2);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('dominant-baseline', 'middle');
    text.setAttribute('fill', 'white');
    text.setAttribute('font-size', '14');
    text.setAttribute('font-weight', 'bold');
    text.textContent = comp.label;
    
    g.appendChild(rect);
    g.appendChild(text);
    svg.appendChild(g);
    
    // Add click event for details
    g.addEventListener('click', () => showComponentDetails(comp));
}

// Draw all connections
function drawAllConnections(svg) {
    // Create a group for all paths
    const pathGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    pathGroup.setAttribute('id', 'connections');
    
    // Draw all possible connections
    const allConnections = [
        ...connections.simple,
        ...connections.complex,
        ...connections.rag,
        ...connections.observability
    ];
    
    // Remove duplicates based on id
    const uniqueConnections = allConnections.filter((conn, index, self) =>
        index === self.findIndex((c) => c.id === conn.id)
    );
    
    uniqueConnections.forEach(conn => {
        const path = createConnectionPath(conn);
        pathGroup.appendChild(path);
    });
    
    // Insert at the beginning of SVG (behind components)
    svg.insertBefore(pathGroup, svg.firstChild.nextSibling.nextSibling);
}

// Create connection path
function createConnectionPath(conn) {
    const fromComp = components[conn.from];
    const toComp = components[conn.to];
    
    if (!fromComp || !toComp) return null;
    
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.classList.add('connection-path');
    path.setAttribute('id', conn.id);
    
    // Calculate path coordinates
    const startX = fromComp.x + fromComp.width;
    const startY = fromComp.y + fromComp.height / 2;
    const endX = toComp.x;
    const endY = toComp.y + toComp.height / 2;
    
    // Create curved path
    const midX = (startX + endX) / 2;
    const d = `M ${startX} ${startY} Q ${midX} ${startY} ${midX} ${(startY + endY) / 2} T ${endX} ${endY}`;
    
    path.setAttribute('d', d);
    path.setAttribute('marker-end', 'url(#arrowhead)');
    
    if (conn.dashed) {
        path.setAttribute('stroke-dasharray', '5,5');
        path.setAttribute('stroke', '#f97316');
        path.setAttribute('opacity', '0.4');
    }
    
    return path;
}

// Setup event listeners
function setupEventListeners() {
    const buttons = document.querySelectorAll('.use-case-btn');
    
    buttons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const useCase = e.currentTarget.dataset.case;
            
            // Remove active class from all buttons
            buttons.forEach(b => b.classList.remove('active'));
            
            if (useCase === 'reset') {
                resetView();
            } else {
                e.currentTarget.classList.add('active');
                activateUseCase(useCase);
            }
        });
    });
}

// Activate use case
function activateUseCase(useCase) {
    currentUseCase = useCase;
    resetHighlights();
    
    // Always keep observability highlighted
    highlightObservability();
    
    // Highlight relevant paths
    const paths = connections[useCase];
    paths.forEach(conn => {
        const path = document.getElementById(conn.id);
        if (path) {
            path.classList.add('active');
            path.setAttribute('marker-end', 'url(#arrowhead-active)');
        }
    });
    
    // Highlight relevant components
    const relevantComponents = new Set();
    paths.forEach(conn => {
        relevantComponents.add(conn.from);
        relevantComponents.add(conn.to);
    });
    
    relevantComponents.forEach(compId => {
        const element = document.querySelector(`[data-id="${compId}"]`);
        if (element) {
            element.classList.add('highlight');
        }
    });
    
    // Update metrics
    updateMetrics(useCase);
    
    // Add log entry
    addLog('success', `Activated ${useCase} use case flow`);
    
    // Simulate processing
    simulateProcessing(useCase);
}

// Reset all highlights
function resetHighlights() {
    document.querySelectorAll('.connection-path').forEach(path => {
        path.classList.remove('active');
        path.setAttribute('marker-end', 'url(#arrowhead)');
    });
    
    document.querySelectorAll('.component-box').forEach(comp => {
        comp.classList.remove('highlight', 'active');
    });
}

// Reset view
function resetView() {
    currentUseCase = null;
    resetHighlights();
    highlightObservability();
    document.querySelectorAll('.use-case-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    addLog('info', 'View reset to default state');
}

// Highlight observability layer
function highlightObservability() {
    const obsBox = document.querySelector('[data-id="observability"]');
    if (obsBox) {
        obsBox.classList.add('highlight');
        const rect = obsBox.querySelector('rect');
        rect.setAttribute('fill-opacity', '1');
        rect.setAttribute('filter', 'drop-shadow(0 0 15px rgba(249, 115, 22, 0.6))');
    }
    
    // Show observability connections
    connections.observability.forEach(conn => {
        const path = document.getElementById(conn.id);
        if (path) {
            path.setAttribute('opacity', '0.4');
        }
    });
}

// Show component details
function showComponentDetails(comp) {
    const panel = document.getElementById('details-panel');
    const content = panel.querySelector('.details-content');
    
    content.innerHTML = `
        <h4 style="color: ${comp.color}; margin-bottom: 8px;">${comp.label}</h4>
        <p>${comp.description}</p>
    `;
    
    panel.classList.add('show');
    
    // Hide after 5 seconds
    setTimeout(() => {
        panel.classList.remove('show');
    }, 5000);
}

// Update metrics based on use case
function updateMetrics(useCase) {
    requestCount++;
    document.getElementById('request-count').textContent = requestCount;
    
    // Simulate different metrics for different use cases
    switch(useCase) {
        case 'simple':
            avgLatency = Math.floor(50 + Math.random() * 50);
            accuracyScore = Math.floor(85 + Math.random() * 10);
            memoryUsage = Math.floor(10 + Math.random() * 20);
            break;
        case 'complex':
            avgLatency = Math.floor(200 + Math.random() * 100);
            accuracyScore = Math.floor(90 + Math.random() * 8);
            memoryUsage = Math.floor(50 + Math.random() * 50);
            break;
        case 'rag':
            avgLatency = Math.floor(150 + Math.random() * 75);
            accuracyScore = Math.floor(92 + Math.random() * 6);
            memoryUsage = Math.floor(30 + Math.random() * 40);
            break;
    }
    
    // Animate metric updates
    animateValue('latency', avgLatency, 'ms');
    animateValue('accuracy', accuracyScore, '%');
    animateValue('memory', memoryUsage, 'MB');
}

// Animate value changes
function animateValue(id, value, suffix) {
    const element = document.getElementById(id);
    const current = parseInt(element.textContent) || 0;
    const increment = (value - current) / 20;
    let step = 0;
    
    const timer = setInterval(() => {
        step++;
        const newValue = Math.round(current + increment * step);
        element.textContent = newValue + suffix;
        
        if (step >= 20) {
            clearInterval(timer);
            element.textContent = value + suffix;
        }
    }, 30);
}

// Add log entry
function addLog(type, message) {
    const container = document.getElementById('log-container');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    
    const timestamp = new Date().toLocaleTimeString();
    const typeLabel = type.toUpperCase();
    entry.textContent = `[${typeLabel}] ${timestamp} - ${message}`;
    
    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;
    
    // Keep only last 10 entries
    while (container.children.length > 10) {
        container.removeChild(container.firstChild);
    }
}

// Simulate processing flow
function simulateProcessing(useCase) {
    const steps = getProcessingSteps(useCase);
    let stepIndex = 0;
    
    const processStep = () => {
        if (stepIndex < steps.length) {
            const step = steps[stepIndex];
            highlightComponent(step.component);
            addLog(step.type || 'info', step.message);
            stepIndex++;
            setTimeout(processStep, 1000);
        }
    };
    
    processStep();
}

// Get processing steps for each use case
function getProcessingSteps(useCase) {
    const steps = {
        simple: [
            { component: 'input', message: 'Receiving user input...' },
            { component: 'complexity', message: 'Analyzing complexity: SIMPLE', type: 'info' },
            { component: 'agentA', message: 'Processing with Agent A...', type: 'success' },
            { component: 'output', message: 'Response generated successfully', type: 'success' }
        ],
        complex: [
            { component: 'input', message: 'Receiving user input...' },
            { component: 'complexity', message: 'Analyzing complexity: COMPLEX', type: 'warning' },
            { component: 'langgraph', message: 'LangGraph reasoning engine routing request...', type: 'info' },
            { component: 'agentB', message: 'Agent B processing analytical tasks...', type: 'success' },
            { component: 'redis', message: 'Storing state in Redis memory...', type: 'info' },
            { component: 'agentA', message: 'Agent A invoking MCP server...', type: 'success' },
            { component: 'mcp', message: 'MCP server connecting to System A...', type: 'info' },
            { component: 'systemA', message: 'System A processing request...', type: 'info' },
            { component: 'output', message: 'Complex response assembled', type: 'success' }
        ],
        rag: [
            { component: 'input', message: 'Receiving RAG query...' },
            { component: 'complexity', message: 'Routing to RAG pipeline...', type: 'info' },
            { component: 'internalData', message: 'Accessing internal data sources...', type: 'info' },
            { component: 'rag', message: 'Performing RAG on internal data...', type: 'success' },
            { component: 'ragas', message: 'RAGAS evaluating response quality...', type: 'warning' },
            { component: 'output', message: 'RAG response delivered', type: 'success' }
        ]
    };
    
    return steps[useCase] || [];
}

// Highlight a specific component temporarily
function highlightComponent(compId) {
    const element = document.querySelector(`[data-id="${compId}"]`);
    if (element) {
        element.classList.add('active');
        setTimeout(() => {
            element.classList.remove('active');
        }, 800);
    }
}

// Start metrics simulation
function startMetricsSimulation() {
    // Simulate random metric updates
    setInterval(() => {
        if (currentUseCase) {
            // Slight random variations
            const latencyVariation = Math.floor(Math.random() * 20 - 10);
            const currentLatency = parseInt(document.getElementById('latency').textContent);
            if (currentLatency > 0) {
                document.getElementById('latency').textContent = 
                    Math.max(10, currentLatency + latencyVariation) + 'ms';
            }
            
            // Memory fluctuation
            const memoryVariation = Math.floor(Math.random() * 5 - 2);
            const currentMemory = parseInt(document.getElementById('memory').textContent);
            if (currentMemory > 0) {
                document.getElementById('memory').textContent = 
                    Math.max(5, currentMemory + memoryVariation) + 'MB';
            }
        }
    }, 3000);
    
    // Add periodic observability logs
    setInterval(() => {
        if (currentUseCase) {
            const messages = [
                'Metrics collected and sent to Azure Monitor',
                'Trace data exported to observability backend',
                'Health check completed - all systems operational',
                'Performance threshold alert: Response time within limits',
                'Memory usage optimized through Redis cache'
            ];
            const randomMessage = messages[Math.floor(Math.random() * messages.length)];
            addLog('info', `[Observability] ${randomMessage}`);
        }
    }, 5000);
}