let currentPcaData = null;

document.addEventListener('DOMContentLoaded', () => {
  runSegmentation();
});

function toggleTheme() {
  document.body.classList.toggle('light-theme');
  if (currentPcaData) drawPcaScatterPlot(currentPcaData);
}

function switchNavTab(tabId) {
  document.querySelectorAll('.view-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));

  document.getElementById(tabId).classList.add('active');

  if (tabId === 'profiles-view') document.getElementById('tab-btn-profiles').classList.add('active');
  if (tabId === 'pca-view') {
    document.getElementById('tab-btn-pca').classList.add('active');
    if (currentPcaData) drawPcaScatterPlot(currentPcaData);
  }
  if (tabId === 'revenue-view') document.getElementById('tab-btn-revenue').classList.add('active');
  if (tabId === 'validation-view') document.getElementById('tab-btn-validation').classList.add('active');
}

function updateKValue() {
  const k = document.getElementById('k-slider').value;
  document.getElementById('k-val').innerText = k;
  runSegmentation();
}

async function runSegmentation() {
  const algo = document.getElementById('algo-select').value;
  const k = parseInt(document.getElementById('k-slider').value, 10);
  const btn = document.getElementById('segment-btn');

  btn.disabled = true;
  btn.innerText = '⚡ Computing Clusters...';

  try {
    const resp = await fetch('/api/segment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ algorithm: algo, n_clusters: k })
    });

    const data = await resp.json();
    currentPcaData = data.pca_projection;

    // Update Silhouette Badge
    document.getElementById('sil-score-badge').innerHTML = `Silhouette Score: <strong>${data.silhouette_score}</strong>`;

    // Render Views
    renderProfiles(data.profiles);
    renderRevenueSimulation(data.revenue_simulation);
    drawPcaScatterPlot(data.pca_projection);
  } catch (e) {
    console.error('Segmentation error:', e);
  } finally {
    btn.disabled = false;
    btn.innerText = '⚡ Run Segmentation & Personalization';
  }
}

function renderProfiles(profiles) {
  const container = document.getElementById('profiles-grid');
  container.innerHTML = profiles.map(p => `
    <div class="profile-card">
      <div class="profile-header">
        <div class="persona-title">${escapeHtml(p.persona)}</div>
        <span class="persona-badge" style="background:${p.color}">${p.percentage}% (${p.count})</span>
      </div>

      <div class="metrics-list">
        <div class="metric-item"><span>Recency</span><strong>${p.mean_recency_days} days</strong></div>
        <div class="metric-item"><span>Frequency</span><strong>${p.mean_frequency_orders} orders</strong></div>
        <div class="metric-item"><span>Monetary</span><strong>$${p.mean_monetary_spend}</strong></div>
        <div class="metric-item"><span>Avg Order</span><strong>$${p.mean_aov}</strong></div>
      </div>

      <div class="action-box">
        <h5>🎯 Marketing Action: ${escapeHtml(p.recommended_action)}</h5>
        <p>${escapeHtml(p.action_description)}</p>
      </div>
    </div>
  `).join('');
}

function renderRevenueSimulation(sim) {
  document.getElementById('rev-base').innerText = `$${sim.baseline_campaign_revenue.toLocaleString()}`;
  document.getElementById('rev-pers').innerText = `$${sim.personalized_campaign_revenue.toLocaleString()}`;
  document.getElementById('rev-lift').innerText = `+$${sim.revenue_lift_dollars.toLocaleString()}`;
  document.getElementById('rev-lift-pct').innerText = `+${sim.revenue_lift_percentage}%`;

  const tbody = document.querySelector('#revenue-table tbody');
  tbody.innerHTML = sim.segment_breakdown.map(s => `
    <tr>
      <td><strong>${escapeHtml(s.persona)}</strong></td>
      <td>${s.customer_count}</td>
      <td>${s.baseline_cr}</td>
      <td><span style="color:#10b981; font-weight:600;">${s.personalized_cr}</span></td>
      <td>$${s.baseline_revenue.toLocaleString()}</td>
      <td>$${s.personalized_revenue.toLocaleString()}</td>
      <td><strong style="color:#10b981;">+$${s.segment_revenue_lift.toLocaleString()}</strong></td>
    </tr>
  `).join('');
}

// Draw 2D PCA Scatter Plot
function drawPcaScatterPlot(pcaData) {
  const canvas = document.getElementById('pca-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const isDark = !document.body.classList.contains('light-theme');
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)';

  // Draw Grid
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;
  for (let x = 0; x < canvas.width; x += 50) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }
  for (let y = 0; y < canvas.height; y += 50) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  }

  const pts = pcaData.points;
  if (!pts || pts.length === 0) return;

  // Find min/max PC values for scaling to canvas width/height
  const pc1s = pts.map(p => p.pc1);
  const pc2s = pts.map(p => p.pc2);

  const minX = Math.min(...pc1s), maxX = Math.max(...pc1s);
  const minY = Math.min(...pc2s), maxY = Math.max(...pc2s);

  const padding = 50;
  const scaleX = (val) => padding + ((val - minX) / (maxX - minX || 1)) * (canvas.width - 2 * padding);
  const scaleY = (val) => canvas.height - (padding + ((val - minY) / (maxY - minY || 1)) * (canvas.height - 2 * padding));

  const colors = ["#10b981", "#ef4444", "#3b82f6", "#eab308", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"];

  // Draw Customer Data Points
  pts.forEach(p => {
    const cx = scaleX(p.pc1);
    const cy = scaleY(p.pc2);

    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fillStyle = colors[p.cluster % colors.length];
    ctx.globalAlpha = 0.7;
    ctx.fill();
    ctx.globalAlpha = 1.0;
  });

  // Draw Centroids
  pcaData.centroids.forEach(c => {
    const cx = scaleX(c.pc1);
    const cy = scaleY(c.pc2);

    ctx.beginPath();
    ctx.arc(cx, cy, 9, 0, Math.PI * 2);
    ctx.fillStyle = colors[c.cluster % colors.length];
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 10px sans-serif';
    ctx.fillText(`C${c.cluster}`, cx - 6, cy + 3);
  });

  // Variance text
  document.getElementById('var-explained-text').innerText = `Variance Explained: PC1 = ${pcaData.variance_explained[0]}%, PC2 = ${pcaData.variance_explained[1]}%`;
}

// Load Validation Curves (Elbow & Silhouette)
async function loadValidationData() {
  const loader = document.getElementById('val-loader');
  const results = document.getElementById('val-results');

  loader.style.display = 'block';
  results.style.display = 'none';

  try {
    const resp = await fetch('/api/validation');
    const data = await resp.json();

    const tbody = document.querySelector('#val-table tbody');
    tbody.innerHTML = data.k_values.map((k, idx) => {
      const isBest = k === data.recommended_k;
      return `
        <tr ${isBest ? 'style="background:rgba(16,185,129,0.15);"' : ''}>
          <td><strong>K = ${k}</strong></td>
          <td><code>${data.wcss[idx]}</code></td>
          <td><code>${data.silhouette_scores[idx]}</code></td>
          <td>${isBest ? '<strong style="color:#10b981;">★ Recommended Optimal K</strong>' : '-'}</td>
        </tr>
      `;
    }).join('');

    results.style.display = 'block';
  } catch (e) {
    console.error('Validation error:', e);
  } finally {
    loader.style.display = 'none';
  }
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}
