/**
 * alert_queue.js
 * Manages the main left pane Alert Queue.
 */

import ApiClient from './api_client.js';

class AlertQueue {
  constructor(entityView) {
    this.entityView = entityView;
    this.tbody = document.getElementById('alerts-tbody');
    
    // Filters
    this.tierFilter = document.getElementById('filter-tier');
    this.classFilter = document.getElementById('filter-class');
    this.btnRefresh = document.getElementById('btn-refresh');
    this.btnStream = document.getElementById('btn-stream');
    this.errorBanner = document.getElementById('error-banner');
    
    this.alertsData = [];
    this.streamConnection = null;
    this.isStreaming = false;
    
    // Bind events
    this.btnRefresh.addEventListener('click', () => this.refresh());
    this.tierFilter.addEventListener('change', () => this.refresh());
    this.classFilter.addEventListener('change', () => this.refresh());
    this.btnStream.addEventListener('click', () => this.toggleStream());
    
    // Initial Load
    this.refresh();
  }

  showError(msg) {
    this.errorBanner.style.display = 'block';
    this.errorBanner.innerText = msg;
  }
  
  hideError() {
    this.errorBanner.style.display = 'none';
  }

  async refresh() {
    this.hideError();
    this.tbody.innerHTML = '<tr><td colspan="6" style="text-align: center;">Loading alerts...</td></tr>';
    
    const params = {
      page: 1,
      page_size: 100
    };
    
    if (this.tierFilter.value) params.risk_tier = this.tierFilter.value;
    if (this.classFilter.value) params.attack_class = this.classFilter.value;
    
    try {
      const response = await ApiClient.fetchAlerts(params);
      this.alertsData = response.alerts || [];
      this.render();
    } catch (error) {
      this.tbody.innerHTML = '<tr><td colspan="6" style="text-align: center;">Failed to load alerts.</td></tr>';
      this.showError(error.message);
    }
  }

  toggleStream() {
    if (this.isStreaming) {
      // Stop
      if (this.streamConnection) this.streamConnection.close();
      this.isStreaming = false;
      this.btnStream.innerText = "Start Stream";
      this.btnStream.style.backgroundColor = "var(--accent-color)";
    } else {
      // Start
      this.hideError();
      this.isStreaming = true;
      this.btnStream.innerText = "Stop Stream";
      this.btnStream.style.backgroundColor = "var(--tier-critical)";
      
      this.streamConnection = ApiClient.connectStream(
        (newAlert) => {
          // Prepend new alert
          this.alertsData.unshift(newAlert);
          // Keep max 100 in view
          if (this.alertsData.length > 100) this.alertsData.pop();
          this.render();
        },
        (error) => {
          this.showError("Stream error or disconnected");
          this.toggleStream(); // auto-stop on error
        }
      );
    }
  }

  render() {
    this.tbody.innerHTML = '';
    
    if (this.alertsData.length === 0) {
      this.tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-secondary);">No alerts found matching criteria.</td></tr>';
      return;
    }
    
    this.alertsData.forEach(alert => {
      const tr = document.createElement('tr');
      
      const time = new Date(alert.timestamp).toLocaleTimeString();
      const entityStr = alert.cold_start_flag 
        ? `${alert.entity_id} <span class="badge cold-start-badge">Cold Start</span>`
        : alert.entity_id;
        
      const tierBadge = `<span class="badge tier-${alert.risk_tier.toLowerCase()}">${alert.risk_tier}</span>`;
      
      tr.innerHTML = `
        <td>${time}</td>
        <td>${tierBadge}</td>
        <td>${alert.risk_score}</td>
        <td>${entityStr}</td>
        <td>${alert.attack_class}</td>
        <td><div style="max-width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${alert.human_readable_explanation}">${alert.human_readable_explanation}</div></td>
      `;
      
      tr.addEventListener('click', () => {
        // Highlight row
        Array.from(this.tbody.children).forEach(r => r.classList.remove('selected'));
        tr.classList.add('selected');
        
        // Load details
        this.entityView.loadDetails(alert.alert_id, alert.entity_id);
      });
      
      this.tbody.appendChild(tr);
    });
  }
}

export default AlertQueue;
