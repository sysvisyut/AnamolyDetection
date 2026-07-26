/**
 * entity_view.js
 * Handles the right pane Details Sidebar: Alert explanations and Entity profile summary.
 */

import ApiClient from './api_client.js';

class EntityView {
  constructor() {
    this.container = document.getElementById('details-pane');
    this.alertExplanationDiv = document.getElementById('alert-explanation');
    this.entityStatusDiv = document.getElementById('entity-status');
    this.entityHistoryDiv = document.getElementById('entity-history');
    
    // UI elements for Explanation
    this.humanReadableText = document.getElementById('human-readable-text');
    this.featureAttributionsContainer = document.getElementById('feature-attributions');
    
    // UI elements for Status
    this.statusEntityId = document.getElementById('status-entity-id');
    this.statusProfileVersion = document.getElementById('status-profile-version');
    this.statusDriftSeverity = document.getElementById('status-drift-severity');
    
    // UI elements for History
    this.historyList = document.getElementById('entity-history-list');
  }

  showEmptyState() {
    this.container.innerHTML = `<div class="empty-state">Select an alert to view details</div>`;
  }

  showLoadingState() {
    this.container.innerHTML = `<div class="empty-state">Loading details...</div>`;
  }
  
  showErrorState(message) {
    this.container.innerHTML = `<div class="empty-state" style="color: var(--tier-critical);">Error: ${message}</div>`;
  }
  
  restoreLayout() {
    // Restore the layout if it was overwritten by empty/loading states
    if (this.container.querySelector('.empty-state')) {
      this.container.innerHTML = `
        <div id="alert-explanation" class="section">
          <h2>Alert Explanation</h2>
          <div id="human-readable-text" class="explanation-text"></div>
          <div id="feature-attributions" class="feature-attributions"></div>
        </div>
        <hr style="border: 1px solid var(--border-color); margin: 20px 0;">
        <div id="entity-status" class="section">
          <h2>Entity Profile Status</h2>
          <div><strong>Entity ID:</strong> <span id="status-entity-id"></span></div>
          <div><strong>Profile Version:</strong> <span id="status-profile-version"></span></div>
          <div><strong>Drift Severity:</strong> <span id="status-drift-severity"></span></div>
        </div>
        <div id="entity-history" class="section">
          <h2>Event History</h2>
          <ul id="entity-history-list" class="entity-history-list"></ul>
        </div>
      `;
      // Re-bind elements
      this.humanReadableText = document.getElementById('human-readable-text');
      this.featureAttributionsContainer = document.getElementById('feature-attributions');
      this.statusEntityId = document.getElementById('status-entity-id');
      this.statusProfileVersion = document.getElementById('status-profile-version');
      this.statusDriftSeverity = document.getElementById('status-drift-severity');
      this.historyList = document.getElementById('entity-history-list');
    }
  }

  async loadDetails(alertId, entityId) {
    this.showLoadingState();
    
    try {
      const [alertDetails, entityStatus, entityHistory] = await Promise.all([
        ApiClient.fetchAlertDetails(alertId),
        ApiClient.fetchEntityStatus(entityId).catch(() => null), // Graceful fallback if entity not found
        ApiClient.fetchEntityHistory(entityId, 10).catch(() => [])
      ]);
      
      this.restoreLayout();
      this.renderExplanation(alertDetails);
      this.renderEntityStatus(entityStatus, entityId);
      this.renderEntityHistory(entityHistory);
      
    } catch (error) {
      console.error("Failed to load details", error);
      this.showErrorState(error.message);
    }
  }
  
  renderExplanation(alert) {
    if (!alert.explanation) {
      this.humanReadableText.innerText = "No explanation available.";
      this.featureAttributionsContainer.innerHTML = "";
      return;
    }
    
    this.humanReadableText.innerText = alert.explanation.human_readable_explanation;
    
    // Render Feature Attributions
    this.featureAttributionsContainer.innerHTML = "<h3>Top Contributing Features</h3>";
    
    const attributions = alert.explanation.feature_attributions || [];
    if (attributions.length === 0) {
      this.featureAttributionsContainer.innerHTML += "<p>None recorded.</p>";
      return;
    }
    
    // Find max magnitude for scaling bars
    const maxScore = Math.max(...attributions.map(a => Math.abs(a.attribution_score)));
    
    attributions.forEach(attr => {
      const barWrapper = document.createElement('div');
      barWrapper.className = 'feature-bar';
      
      const label = document.createElement('div');
      label.className = 'feature-label';
      label.title = attr.human_label;
      label.innerText = attr.human_label || attr.feature_name;
      
      const barContainer = document.createElement('div');
      barContainer.className = 'bar-container';
      
      const barFill = document.createElement('div');
      barFill.className = `bar-fill ${attr.direction === 'toward_anomaly' ? 'toward-anomaly' : 'toward-normal'}`;
      
      // Calculate width percentage relative to max score
      const percentage = maxScore > 0 ? (Math.abs(attr.attribution_score) / maxScore) * 50 : 0;
      barFill.style.width = `${percentage}%`;
      
      barContainer.appendChild(barFill);
      
      const val = document.createElement('div');
      val.className = 'feature-value';
      val.innerText = parseFloat(attr.attribution_score).toFixed(2);
      
      barWrapper.appendChild(label);
      barWrapper.appendChild(barContainer);
      barWrapper.appendChild(val);
      
      this.featureAttributionsContainer.appendChild(barWrapper);
    });
  }

  renderEntityStatus(status, entityIdFallback) {
    if (!status) {
      this.statusEntityId.innerText = entityIdFallback;
      this.statusProfileVersion.innerText = "Unknown";
      this.statusDriftSeverity.innerText = "Unknown";
      return;
    }
    
    let entityHtml = status.entity_id;
    if (status.is_cold_start) {
      entityHtml += ' <span class="badge cold-start-badge">Cold Start</span>';
    }
    
    this.statusEntityId.innerHTML = entityHtml;
    this.statusProfileVersion.innerText = `v${status.profile_version}`;
    this.statusDriftSeverity.innerText = status.drift_severity;
  }

  renderEntityHistory(history) {
    this.historyList.innerHTML = "";
    
    if (!history || history.length === 0) {
      this.historyList.innerHTML = "<li>No recent history</li>";
      return;
    }
    
    history.forEach(entry => {
      const li = document.createElement('li');
      li.className = 'entity-history-item';
      
      const time = new Date(entry.timestamp).toLocaleString();
      let eventType = "Event";
      
      // Try to extract useful info depending on schema
      if (entry.event_snapshot && entry.event_snapshot.resource_accessed) {
        eventType = `Accessed ${entry.event_snapshot.resource_accessed}`;
      }
      
      li.innerHTML = `
        <span>${eventType}</span>
        <span style="color: var(--text-secondary); font-size: 0.8em;">${time}</span>
      `;
      this.historyList.appendChild(li);
    });
  }
}

export default EntityView;
