// Regime Lab website: one plain script. It calls the Python model at /api/* and draws the results with Plotly.
'use strict';

// ================================================================ shared helpers
// Shared helpers: API calls, loading states, formatting and the Plotly theme.

const cache = new Map();
const inFlight = new Map();

async function api(path, params = {}, { useCache = true } = {}) {
  const qs = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== null && v !== undefined && v !== ''));
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;
  const url = `/api/${cleanPath}${qs.toString() ? '?' + qs : ''}`;
  if (useCache && cache.has(url)) return cache.get(url);
  if (inFlight.has(url)) return inFlight.get(url);

  const fetchPromise = (async () => {
    try {
      const t0 = performance.now();
      const res = await fetch(url);
      const ms = performance.now() - t0;
      if (!res.ok) {
        let msg = `${res.status} ${res.statusText}`;
        try { const j = await res.json(); msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (e) { /* keep status text */ }
        throw new Error(msg);
      }
      const data = await res.json();
      const out = { data, url, ms, server: Number(res.headers.get('X-Compute-Ms')) };
      if (useCache) cache.set(url, out);
      return out;
    } finally {
      inFlight.delete(url);
    }
  })();

  inFlight.set(url, fetchPromise);
  return fetchPromise;
}

function computeChip(el, call) {
  if (!el || !call) return;
  el.innerHTML = `<span>Computed by</span> <code>GET ${esc(call.url.split('?')[0])}</code> <span>in</span> <b>${fmtMs(call.server || call.ms)}</b>`;
  el.title = call.url;
}

function fmtMs(ms) { return ms >= 1000 ? (ms / 1000).toFixed(1) + ' s' : Math.round(ms) + ' ms'; }

async function withBusy(el, label, fn) {
  el.classList.add('busy');
  el.dataset.busy = label;
  try { return await fn(); }
  catch (e) { toast(e.message, true); throw e; }
  finally { el.classList.remove('busy'); }
}

let toastTimer;
function toast(msg, isErr = false) {
  let t = document.querySelector('.toast');
  if (!t) { t = document.createElement('div'); t.className = 'toast'; t.setAttribute('role', 'status'); document.body.appendChild(t); }
  t.textContent = msg;
  t.classList.toggle('err', isErr);
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, isErr ? 7000 : 3500);
}

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (x, d = 0) => (x == null ? '—' : (x * 100).toFixed(d) + '%');
const spct = (x, d = 1) => (x == null ? '—' : (x > 0 ? '+' : '') + (x * 100).toFixed(d) + '%');
const num = (x, d = 2) => (x == null ? '—' : Number(x).toFixed(d));
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function segmented(el, value, onChange) {
  const set = (v) => $$('button', el).forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === String(v))));
  set(value);
  el.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    set(b.dataset.v);
    onChange(b.dataset.v);
  });
  return { set };
}

function rangeField(input, out, fmt = (v) => v) {
  const upd = () => { out.textContent = fmt(input.value); };
  input.addEventListener('input', upd);
  upd();
}

// ---------------------------------------------------------------- Plotly theme
const INK = { ink: '#132033', ink2: '#4a5a6e', ink3: '#7a8798', line: '#dde3ea', grid: '#edf0f4' };

function layout(extra = {}) {
  const base = {
    font: { family: '"IBM Plex Sans", Segoe UI, system-ui, sans-serif', size: 12.5, color: INK.ink2 },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 52, r: 16, t: 12, b: 40 },
    hoverlabel: { bgcolor: '#132033', bordercolor: '#132033', font: { color: '#fff', family: '"IBM Plex Sans", system-ui, sans-serif', size: 13 } },
    legend: { orientation: 'h', y: 1.08, x: 0, font: { size: 12.5 } },
    xaxis: { gridcolor: INK.grid, zerolinecolor: INK.line, linecolor: INK.line, tickcolor: INK.line, automargin: true },
    yaxis: { gridcolor: INK.grid, zerolinecolor: INK.line, linecolor: INK.line, tickcolor: INK.line, automargin: true },
    hovermode: 'closest',
  };
  const out = { ...base, ...extra };
  for (const k of Object.keys(extra)) {
    if (/^[xy]axis\d*$/.test(k)) out[k] = { ...base.xaxis, ...extra[k] };
  }
  return out;
}

const plotConfig = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'] };

function plot(el, traces, lay = {}) {
  if (!el) return Promise.resolve(null);
  const arr = Array.isArray(traces) ? traces : [traces];
  const safeTraces = arr.map((t) => {
    if (!t) return t;
    if (t.type === 'scattergl') return { ...t, type: 'scatter' };
    return t;
  });
  return Plotly.react(el, safeTraces, layout(lay), plotConfig);
}

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

// Shaded background bands for runs of the same state (for time-series charts).
function stateBands(dates, states, colors, alpha = 0.16, yref = 'paper') {
  const shapes = [];
  let s0 = 0;
  for (let i = 1; i <= states.length; i++) {
    if (i === states.length || states[i] !== states[s0]) {
      shapes.push({ type: 'rect', xref: 'x', yref, x0: dates[s0], x1: dates[Math.min(i, dates.length - 1)], y0: 0, y1: 1,
        fillcolor: hexA(colors[states[s0]], alpha), line: { width: 0 }, layer: 'below' });
      s0 = i;
    }
  }
  return shapes;
}

function resizeCanvas(cv, h) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth;
  cv.style.height = h + 'px';
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  const c = cv.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { c, w, h };
}

// ================================================================ view: overview
const VIEW_overview = (() => {
  let reg = null;

  return {
    mount(root) {
      root.innerHTML = `
        <div class="hero" id="ov-hero">
          <div>
            <div class="asof" id="ov-asof">Reading the latest month…</div>
            <h2 id="ov-title">What mood is the market in?</h2>
            <p>A Hidden Markov Model trained on every month of the S&amp;P 500 since 1871 estimates which hidden regime the market is in right now, using only prices up to the latest month.</p>
          </div>
          <div class="belief" id="ov-belief"></div>
        </div>

        <div class="panel" id="ov-ribbon-panel">
          <div class="panel-head">
            <h3>155 years in one strip</h3>
            <span class="compute" id="ov-chip"></span>
          </div>
          <p class="sub">Each sliver is one month, coloured by the regime the model decoded. Hover to read any month.</p>
          <div class="ribbon-wrap">
            <canvas class="ribbon" id="ov-ribbon" aria-label="Regime of every month from 1871 to today"></canvas>
            <div class="ribbon-axis" id="ov-axis"></div>
            <div class="ribbon-read" id="ov-read">&nbsp;</div>
          </div>
        </div>

        <div class="panel">
          <h3>The project in numbers</h3>
          <div class="kpis" id="ov-kpis"></div>
        </div>

        <div class="grid g2">
          <div class="panel">
            <h3>The problem</h3>
            <p class="note" style="margin:0">Investors face two questions with no labelled answers. <b>Which stocks behave alike?</b> Sector labels say what a company sells, not how its stock moves, so diversifying by sector can still double up on the same risk. <b>What mood is the market in?</b> Bull, correction and crisis regimes are never announced; we only see prices, and the regime persists over time.</p>
            <p class="note" style="margin:0">The first question is a clustering problem. The second is exactly what a Hidden Markov Model is built for: a hidden state that follows a Markov chain and emits the returns we observe.</p>
          </div>
          <div class="panel">
            <h3>Datasets</h3>
            <div class="scroll-x"><table id="ov-data" class="wrap"></table></div>
          </div>
        </div>

        <div class="panel">
          <h3>How the pipeline runs</h3>
          <div class="steps">
            <a class="step" href="#clustering"><span class="n">Step 1</span><b>Fingerprint</b><span>6 behavioural features per stock</span></a>
            <a class="step" href="#clustering"><span class="n">Step 2</span><b>Cluster</b><span>K-Means, GMM, Ward, DBSCAN</span></a>
            <a class="step" href="#hmm"><span class="n">Step 3</span><b>Fit the HMM</b><span>Baum-Welch on monthly returns</span></a>
            <a class="step" href="#lab"><span class="n">Step 4</span><b>Decode</b><span>Viterbi path and forward filter</span></a>
            <a class="step" href="#compare"><span class="n">Step 5</span><b>Compare</b><span>Metrics, stability, VIX check</span></a>
            <a class="step" href="#backtest"><span class="n">Step 6</span><b>Test</b><span>Out-of-sample backtest</span></a>
            <a class="step" href="#fusion"><span class="n">Step 7</span><b>Fuse</b><span>Which cluster to hold when</span></a>
          </div>
        </div>`;
      load(root);
      window.addEventListener('resize', () => reg && drawRibbon());
    },
    onShow() { reg && drawRibbon(); },
  };

  async function load(root) {
    const hero = $('#ov-hero', root);
    const call = await withBusy(hero, 'Loading the model…', () => api('overview'));
    const d = call.data;
    const now = d.now;
    const i = now.names.indexOf(now.state);
    const dt = new Date(now.date + '-01T00:00').toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
    $('#ov-asof', root).textContent = `Latest month in the data: ${dt}`;
    $('#ov-title', root).innerHTML = `The market is in a <span style="color:${now.colors[i]}">${esc(now.state)}</span> regime`;
    $('#ov-belief', root).innerHTML =
      `<h4>Belief this month</h4>` + now.names.map((n, j) => row(n, now.filtered[j], now.colors[j])).join('') +
      `<h4>Forecast for next month</h4>` + now.names.map((n, j) => row(n, now.next_month[j], now.colors[j])).join('');

    const h = d.headline;
    $('#ov-kpis', root).innerHTML = [
      [h.stocks, 'S&P 500 stocks clustered into ' + h.clusters.length + ' personalities'],
      [h.years + ' yrs', `${h.months.toLocaleString()} monthly returns decoded`],
      [h.regimes.length, 'hidden regimes: ' + h.regimes.map((r) => `${r.name} ${pct(r.share)}`).join(', ')],
      [`${pct(h.drawdown_bh)} → ${pct(h.drawdown_hmm)}`, 'worst loss, buy & hold vs HMM strategy, 1990–2026'],
      [`${num(h.sharpe_bh)} → ${num(h.sharpe_hmm)}`, 'Sharpe ratio out of sample'],
    ].map(([v, l]) => `<div class="kpi"><span class="v">${v}</span><span class="l">${esc(l)}</span></div>`).join('');

    $('#ov-data', root).innerHTML = '<tr><th>Data</th><th>Span</th><th>Size</th><th>Used for</th></tr>' +
      d.datasets.map((x) => `<tr><td><b>${x.id}</b> ${esc(x.name)}<br><small style="color:var(--ink-3)">${esc(x.source)}</small></td><td>${esc(x.span)}</td><td>${esc(x.size)}</td><td>${esc(x.used_for)}</td></tr>`).join('');

    const rcall = await api('hmm', { n_states: 3, start: 1871, end: 2026 });
    reg = rcall.data;
    computeChip($('#ov-chip', root), rcall);
    drawRibbon();
    const cv = $('#ov-ribbon', root);
    cv.addEventListener('pointermove', (e) => readAt(e, cv, root));
    cv.addEventListener('pointerleave', () => { $('#ov-read', root).innerHTML = '&nbsp;'; drawRibbon(); });
  }

  function row(name, p, color) {
    return `<div class="belief-row"><span>${esc(name)}</span><span class="bar"><span style="width:${(p * 100).toFixed(1)}%;background:${color}"></span></span><b>${pct(p)}</b></div>`;
  }

  function drawRibbon(hover = -1) {
    const cv = document.getElementById('ov-ribbon');
    if (!cv || !cv.clientWidth) return;
    const { c, w, h } = resizeCanvas(cv, 96);
    const s = reg.series;
    const n = s.state.length;
    const bw = w / n;
    for (let i = 0; i < n; i++) {
      c.fillStyle = reg.colors[s.state[i]];
      c.fillRect(i * bw, 0, bw + 0.6, h);
    }
    // log price line drawn over the strip
    const lp = s.price.map((p) => Math.log(p));
    const lo = Math.min(...lp), hi = Math.max(...lp);
    c.strokeStyle = 'rgba(19,32,51,0.85)';
    c.lineWidth = 1.5;
    c.beginPath();
    lp.forEach((v, i) => { const x = i * bw, y = h - 8 - (h - 16) * (v - lo) / (hi - lo); i ? c.lineTo(x, y) : c.moveTo(x, y); });
    c.stroke();
    if (hover >= 0) {
      c.fillStyle = 'rgba(255,255,255,0.9)';
      c.fillRect(hover * bw - 1, 0, 2, h);
    }
    const axis = document.getElementById('ov-axis');
    const years = [];
    for (let y = 1880; y <= 2020; y += 20) years.push(y);
    axis.innerHTML = years.map((y) => {
      const i = s.dates.indexOf(y + '-01');
      return i < 0 ? '' : `<span style="left:${(i / n * 100).toFixed(2)}%">${y}</span>`;
    }).join('');
  }

  function readAt(e, cv, root) {
    const r = cv.getBoundingClientRect();
    const s = reg.series;
    const i = Math.max(0, Math.min(s.state.length - 1, Math.floor((e.clientX - r.left) / r.width * s.state.length)));
    drawRibbon(i);
    const dt = new Date(s.dates[i] + '-01T00:00').toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
    const st = reg.names[s.state[i]];
    $('#ov-read', root).innerHTML = `<b>${dt}</b> · S&amp;P ${s.price[i].toLocaleString()} · return ${(s.ret[i] * 100).toFixed(2)}% · decoded <b style="color:${reg.colors[s.state[i]]}">${st}</b>`;
  }
})();

// ================================================================ view: clustering
const VIEW_clustering = (() => {
  const state = { algo: 'kmeans', k: 4, eps: '', min_samples: 12, linkage: 'ward', covariance: 'full' };
  let result = null;
  let highlight = null;
  let tickers = [];
  const SHORT = { ann_return: 'Return', ann_vol: 'Volatility', beta: 'Beta', idio_vol: 'Own risk', max_drawdown: 'Drawdown', skew: 'Skew' };

  return {
    mount(root) {
      root.innerHTML = `
        <div class="view-head">
          <div>
            <h2>Clustering studio</h2>
            <p>Each stock is described by six numbers computed from five years of daily returns. Pick an algorithm and the server clusters all 477 stocks live.</p>
          </div>
          <span class="rubric">Rubric: implementation of clustering, 4 marks</span>
        </div>

        <div class="controls">
          <div class="field"><span>Algorithm</span>
            <div class="seg" id="cl-algo">
              <button data-v="kmeans">K-Means</button><button data-v="gmm">Gaussian mixture</button>
              <button data-v="agglomerative">Agglomerative</button><button data-v="dbscan">DBSCAN</button>
            </div>
          </div>
          <label class="field" id="cl-k-f"><span>Clusters k = <output id="cl-k-o"></output></span><input type="range" id="cl-k" min="2" max="10" value="4"></label>
          <label class="field" id="cl-link-f" hidden><span>Linkage</span><select id="cl-link"><option value="ward">Ward</option><option value="complete">Complete</option><option value="average">Average</option></select></label>
          <label class="field" id="cl-cov-f" hidden><span>Covariance</span><select id="cl-cov"><option value="full">Full</option><option value="diag">Diagonal</option><option value="tied">Tied</option><option value="spherical">Spherical</option></select></label>
          <label class="field" id="cl-eps-f" hidden><span>Radius ε = <output id="cl-eps-o"></output></span><input type="range" id="cl-eps" min="0" max="40" value="0"></label>
          <label class="field" id="cl-ms-f" hidden><span>Min neighbours = <output id="cl-ms-o"></output></span><input type="range" id="cl-ms" min="3" max="30" value="12"></label>
          <span class="spacer"></span>
          <div class="field search"><span>Find a stock</span>
            <input type="text" id="cl-find" placeholder="Ticker, e.g. KO or NVDA" autocomplete="off" aria-label="Find a stock by ticker">
            <div class="suggest" id="cl-suggest" hidden></div>
          </div>
        </div>

        <div class="grid g-main">
          <div class="panel" id="cl-map-panel">
            <div class="panel-head"><h3>Stock map</h3><span class="compute" id="cl-chip"></span></div>
            <p class="sub" id="cl-map-sub">The six features squeezed onto two principal components. Right means riskier, up means higher return. Click a dot to open that stock.</p>
            <div class="plot tall" id="cl-scatter"></div>
          </div>
          <div class="panel" id="cl-side">
            <h3>Quality of this clustering</h3>
            <div class="kpis" id="cl-metrics"></div>
            <div id="cl-extra"></div>
            <div id="cl-legend" class="chips"></div>
          </div>
        </div>

        <div class="grid g2">
          <div class="panel">
            <h3>Cluster fingerprints</h3>
            <p class="sub">Colour shows how far each cluster sits from the average stock (z-score). Numbers are the real averages.</p>
            <div class="plot" id="cl-heat"></div>
          </div>
          <div class="panel">
            <h3>Sectors inside each cluster</h3>
            <p class="sub">If clusters just copied sectors, every bar would be one colour. They do not.</p>
            <div class="plot" id="cl-sector"></div>
          </div>
        </div>

        <div class="panel">
          <h3>The personalities</h3>
          <div class="grid g4" id="cl-cards"></div>
        </div>

        <div class="panel" id="cl-k-panel">
          <div class="panel-head"><h3>Choosing k</h3><span class="compute" id="cl-k-chip"></span></div>
          <p class="sub" id="cl-k-reason">Four criteria computed for k = 2 to 10.</p>
          <div class="grid g4" id="cl-kplots"><div class="plot short"></div><div class="plot short"></div><div class="plot short"></div><div class="plot short"></div></div>
        </div>

        <div class="drawer" id="cl-drawer" hidden role="dialog" aria-label="Stock details"></div>`;

      segmented($('#cl-algo', root), state.algo, (v) => { state.algo = v; syncFields(root); run(root); });
      const k = $('#cl-k', root);
      rangeField(k, $('#cl-k-o', root));
      k.addEventListener('change', () => { state.k = +k.value; run(root); });
      const eps = $('#cl-eps', root);
      rangeField(eps, $('#cl-eps-o', root), (v) => (+v === 0 ? 'auto' : (0.5 + v * 0.1).toFixed(1)));
      eps.addEventListener('change', () => { state.eps = +eps.value === 0 ? '' : (0.5 + eps.value * 0.1).toFixed(1); run(root); });
      const ms = $('#cl-ms', root);
      rangeField(ms, $('#cl-ms-o', root));
      ms.addEventListener('change', () => { state.min_samples = +ms.value; run(root); });
      $('#cl-link', root).addEventListener('change', (e) => { state.linkage = e.target.value; run(root); });
      $('#cl-cov', root).addEventListener('change', (e) => { state.covariance = e.target.value; run(root); });
      setupSearch(root);
      syncFields(root);
      run(root);
      loadK(root);
      api('stocks').then((c) => { tickers = c.data; });
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          const dr = $('#cl-drawer', root);
          if (dr && !dr.hidden) {
            dr.hidden = true;
            highlight = null;
            if (result) drawScatter(root);
          }
        }
      });
    },
  };

  function syncFields(root) {
    const a = state.algo;
    $('#cl-k-f', root).hidden = a === 'dbscan';
    $('#cl-link-f', root).hidden = a !== 'agglomerative';
    $('#cl-cov-f', root).hidden = a !== 'gmm';
    $('#cl-eps-f', root).hidden = a !== 'dbscan';
    $('#cl-ms-f', root).hidden = a !== 'dbscan';
  }

  async function run(root) {
    const params = { algo: state.algo, k: state.k };
    if (state.algo === 'dbscan') Object.assign(params, { eps: state.eps, min_samples: state.min_samples });
    if (state.algo === 'agglomerative') params.linkage = state.linkage;
    if (state.algo === 'gmm') params.covariance = state.covariance;
    const call = await withBusy($('#cl-map-panel', root), 'Clustering 477 stocks…', () => api('cluster', params));
    result = call.data;
    computeChip($('#cl-chip', root), call);
    drawScatter(root);
    drawSide(root);
    drawHeat(root);
    drawSector(root);
    drawCards(root);
  }

  function drawScatter(root) {
    const p = result.points;
    const traces = result.order.map((name) => {
      const idx = p.cluster.map((c, i) => (c === name ? i : -1)).filter((i) => i >= 0);
      return {
        type: 'scatter', mode: 'markers', name, x: idx.map((i) => p.x[i]), y: idx.map((i) => p.y[i]),
        customdata: idx.map((i) => p.ticker[i]),
        text: idx.map((i) => `<b>${p.ticker[i]}</b> · ${p.sector[i]}<br>${name}<br>return ${(p.ret[i] * 100).toFixed(1)}%/yr · vol ${(p.vol[i] * 100).toFixed(0)}% · β ${p.beta[i]}`),
        hovertemplate: '%{text}<extra></extra>',
        marker: { color: result.colors[name], size: 8, line: { color: '#fff', width: 0.8 }, opacity: 0.9 },
      };
    });
    const cen = result.extras.centroids_2d;
    if (cen) {
      traces.push({ type: 'scatter', mode: 'markers', name: 'Centroids', x: cen.map((c) => c[0]), y: cen.map((c) => c[1]), hoverinfo: 'skip',
        marker: { symbol: 'diamond', size: 15, color: '#132033', line: { color: '#fff', width: 2 } } });
    }
    const ann = [];
    if (highlight) {
      const i = p.ticker.indexOf(highlight);
      if (i >= 0) ann.push({ x: p.x[i], y: p.y[i], text: `<b>${highlight}</b>`, showarrow: true, arrowhead: 0, ax: 28, ay: -28, font: { size: 14, color: INK.ink }, bgcolor: '#fff', bordercolor: INK.ink, borderpad: 3 });
    }
    const el = $('#cl-scatter', root);
    plot(el, traces, {
      xaxis: { title: { text: 'PC1: more risk →', font: { size: 12.5 } }, zeroline: false },
      yaxis: { title: { text: 'PC2: more return →', font: { size: 12.5 } }, zeroline: false },
      annotations: ann, margin: { l: 56, r: 12, t: 30, b: 46 },
    });
    if (el && typeof el.removeAllListeners === 'function') {
      try { el.removeAllListeners('plotly_click'); } catch (e) {}
    }
    if (el && typeof el.on === 'function') {
      el.on('plotly_click', (ev) => {
        const t = ev && ev.points && ev.points[0] && ev.points[0].customdata;
        if (t) openStock(root, t);
      });
    }
    $('#cl-map-sub', root).textContent = `The six features squeezed onto two principal components (${pct(result.pca_variance[0] + result.pca_variance[1])} of the variance). Right means riskier, up means higher return. Click a dot to open that stock.`;
  }

  function drawSide(root) {
    const m = result.metrics;
    $('#cl-metrics', root).innerHTML = [
      [num(m.silhouette, 3), 'Silhouette (higher is better)'],
      [num(m.davies_bouldin, 2), 'Davies-Bouldin (lower is better)'],
      [m.calinski_harabasz == null ? '—' : Math.round(m.calinski_harabasz), 'Calinski-Harabasz (higher is better)'],
      [num(m.ari_vs_sector, 3), 'Agreement with GICS sectors (ARI)'],
    ].map(([v, l]) => `<div class="kpi"><span class="v">${v}</span><span class="l">${l}</span></div>`).join('');
    const x = result.extras;
    let html = '';
    if (x.verification) {
      const v = x.verification;
      html = `<div class="verify"><span>Our NumPy K-Means: inertia <b>${num(v.inertia_scratch, 2)}</b></span><span>scikit-learn: <b>${num(v.inertia_sklearn, 2)}</b></span><span>Label agreement (ARI) <b>${num(v.ari_vs_sklearn, 3)}</b> ${v.ari_vs_sklearn > 0.999 ? '<span class="ok-tag">identical</span>' : ''}</span><span>Converged in <b>${v.iterations}</b> iterations</span></div>`;
    } else if (x.eps_used) {
      html = `<div class="verify"><span>ε used <b>${num(x.eps_used, 2)}</b> (auto knee ${num(x.eps_auto, 2)})</span><span><b>${m.n_noise}</b> stocks flagged as outliers</span></div><div class="plot short" id="cl-kd"></div>`;
    } else if (x.bic) {
      html = `<div class="verify"><span>BIC <b>${Math.round(x.bic)}</b></span><span>Mean membership confidence <b>${pct(x.mean_confidence)}</b></span></div>`;
    } else {
      html = `<div class="verify"><span>${esc(result.algo_label)} with ${esc(result.params.linkage)} linkage</span></div>`;
    }
    html += `<p class="note" style="margin:0">Computed in ${result.metrics.time_ms} ms on the server.</p>`;
    $('#cl-extra', root).innerHTML = html;
    if (x.k_distance) {
      plot($('#cl-kd', root), [{ y: x.k_distance, type: 'scatter', mode: 'lines', line: { color: '#3d8bfd', width: 2 }, hovertemplate: '%{y:.2f}<extra></extra>' }],
        { shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: x.eps_used, y1: x.eps_used, line: { color: '#ff6b3d', dash: 'dash' } }],
          xaxis: { title: { text: 'stocks sorted by distance to 12th neighbour' }, showticklabels: false }, yaxis: { title: { text: 'distance' } }, margin: { l: 46, r: 8, t: 8, b: 34 } });
    }
    $('#cl-legend', root).innerHTML = result.clusters.map((c) => `<span class="chip" style="cursor:default"><i class="dot" style="background:${c.color};margin:0"></i>${esc(c.name)} · ${c.n}</span>`).join('');
  }

  function drawHeat(root) {
    const f = result.features;
    const cl = result.clusters;
    const fmtF = (k, v) => (['ann_return', 'ann_vol', 'idio_vol', 'max_drawdown'].includes(k) ? (v * 100).toFixed(0) + '%' : v.toFixed(2));
    plot($('#cl-heat', root), [{
      type: 'heatmap', z: cl.map((c) => f.map((k) => c.z[k])), x: f.map((k) => SHORT[k]), y: cl.map((c) => c.name),
      text: cl.map((c) => f.map((k) => fmtF(k, c.means[k]))), texttemplate: '%{text}', textfont: { size: 13 },
      colorscale: [[0, '#2b63d9'], [0.5, '#f5f7fa'], [1, '#e8590c']], zmid: 0, zmin: -2, zmax: 2, showscale: false,
      hovertemplate: '%{y}<br>%{x}: %{text}<br>z = %{z:.2f}<extra></extra>', xgap: 2, ygap: 2,
    }], { yaxis: { autorange: 'reversed', gridcolor: 'rgba(0,0,0,0)' }, xaxis: { gridcolor: 'rgba(0,0,0,0)', tickangle: 0 }, margin: { l: 110, r: 8, t: 8, b: 40 } });
  }

  function drawSector(root) {
    const s = result.sector_mix;
    plot($('#cl-sector', root), result.order.map((n) => ({
      type: 'bar', orientation: 'h', name: n, y: s.sectors, x: s.shares[n], marker: { color: result.colors[n], line: { color: '#fff', width: 1 } },
      hovertemplate: `%{y}: %{x:.0%} ${n}<extra></extra>`,
    })), { barmode: 'stack', xaxis: { tickformat: '.0%', range: [0, 1] }, yaxis: { autorange: 'reversed' }, margin: { l: 160, r: 8, t: 8, b: 70 }, legend: { orientation: 'h', y: -0.12, yanchor: 'top' }, height: 420 });
  }

  function drawCards(root) {
    $('#cl-cards', root).innerHTML = result.clusters.map((c) => `
      <div style="display:flex;flex-direction:column;gap:8px;padding:14px;border-radius:var(--r-sm);background:var(--soft)">
        <div style="display:flex;align-items:center;gap:8px"><i class="dot" style="background:${c.color};margin:0;width:12px;height:12px"></i><b style="font-size:16px">${esc(c.name)}</b></div>
        <span style="font:700 28px/1 var(--display)">${c.n} <small style="font:400 13px var(--body);color:var(--ink-3)">stocks</small></span>
        <span class="note">return ${pct(c.means.ann_return, 1)}/yr · vol ${pct(c.means.ann_vol)} · β ${num(c.means.beta)} · drawdown ${pct(c.means.max_drawdown)}</span>
        <span class="note">Mostly ${Object.keys(c.top_sectors).slice(0, 2).map(esc).join(', ')}</span>
        <div class="chips">${c.examples.slice(0, 6).map((t) => `<button class="chip" data-t="${esc(t)}">${esc(t)}</button>`).join('')}</div>
      </div>`).join('');
    $('#cl-cards', root).querySelectorAll('[data-t]').forEach((b) => b.addEventListener('click', () => openStock(root, b.dataset.t)));
  }

  async function loadK(root) {
    const call = await withBusy($('#cl-k-panel', root), 'Scoring k = 2…10…', () => api('cluster/select-k'));
    const d = call.data;
    computeChip($('#cl-k-chip', root), call);
    $('#cl-k-reason', root).textContent = d.reason;
    const ks = d.rows.map((r) => r.k);
    const specs = [['inertia', 'Elbow: within-cluster spread ↓'], ['silhouette', 'Silhouette ↑'], ['davies_bouldin', 'Davies-Bouldin ↓'], ['gmm_bic', 'GMM BIC ↓']];
    const plots = $('#cl-kplots', root).children;
    specs.forEach(([key, title], i) => {
      plot(plots[i], [{ x: ks, y: d.rows.map((r) => r[key]), mode: 'lines+markers', line: { color: '#2b63d9', width: 2 }, marker: { size: 7 }, hovertemplate: 'k=%{x}: %{y:.3f}<extra></extra>' }], {
        title: { text: title, font: { size: 13, color: INK.ink }, x: 0, y: 0.98 }, margin: { l: 44, r: 8, t: 30, b: 30 },
        shapes: [{ type: 'line', x0: d.chosen, x1: d.chosen, yref: 'paper', y0: 0, y1: 1, line: { color: '#ff6b3d', dash: 'dot', width: 1.5 } }],
        xaxis: { dtick: 1 },
      });
    });
  }

  function setupSearch(root) {
    const inp = $('#cl-find', root), box = $('#cl-suggest', root);
    inp.addEventListener('input', () => {
      const q = inp.value.trim().toUpperCase();
      if (!q) { box.hidden = true; return; }
      const hits = tickers.filter((t) => t.ticker.startsWith(q)).slice(0, 8);
      box.innerHTML = hits.length ? hits.map((t) => `<button data-t="${t.ticker}"><span>${t.ticker}</span><small>${esc(t.sector)} · ${esc(t.cluster)}</small></button>`).join('') : '<div class="note" style="padding:8px 12px">Not in the 477-stock universe</div>';
      box.hidden = false;
    });
    inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') { const b = box.querySelector('button'); if (b) b.click(); } });
    box.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; box.hidden = true; inp.value = b.dataset.t; openStock(root, b.dataset.t); });
    document.addEventListener('click', (e) => { if (!e.target.closest('.search')) box.hidden = true; });
  }

  async function openStock(root, t) {
    highlight = t;
    if (result) drawScatter(root);
    const dr = $('#cl-drawer', root);
    dr.hidden = false;
    dr.innerHTML = '<p class="note">Loading…</p>';
    let call;
    try { call = await api('stocks/' + encodeURIComponent(t)); } catch (e) { toast(e.message, true); dr.hidden = true; return; }
    const d = call.data;
    const labels = { ann_return: 'Annual return', ann_vol: 'Volatility', beta: 'Beta', idio_vol: 'Stock-specific vol', max_drawdown: 'Max drawdown', skew: 'Skewness' };
    const fmtF = (k, v) => (['ann_return', 'ann_vol', 'idio_vol', 'max_drawdown'].includes(k) ? (v * 100).toFixed(1) + '%' : v.toFixed(2));
    dr.innerHTML = `
      <div class="drawer-head"><div><h3>${esc(d.ticker)}</h3><p class="note" style="margin:6px 0 0">${esc(d.sector)}</p></div><button class="btn" id="dr-close">Close</button></div>
      <div><span class="chip" style="cursor:default"><i class="dot" style="background:${d.color};margin:0"></i>${esc(d.cluster)} cluster (K-Means, k = 4)</span></div>
      <div class="plot short" id="dr-price"></div>
      <h4 style="margin:0">Fingerprint vs all 477 stocks</h4>
      ${Object.keys(labels).map((k) => `<div class="pct-row"><span>${labels[k]}</span><span class="bar"><span style="width:${(d.percentiles[k] * 100).toFixed(0)}%;background:${d.color}"></span></span><span>${fmtF(k, d.features[k])}</span></div>`).join('')}
      <p class="note" style="margin:0">Bars show the percentile rank among the 477 stocks.</p>
      <h4 style="margin:0">Behaves most like</h4>
      <div class="chips">${d.neighbours.map((n) => `<button class="chip" data-t="${n.ticker}">${n.ticker} <small style="color:var(--ink-3)">${esc(n.sector)}</small></button>`).join('')}</div>
      <p class="note" style="margin:0">Nearest neighbours in the six-feature space, often from different sectors.</p>`;
    $('#dr-close', dr).addEventListener('click', () => {
      dr.hidden = true;
      highlight = null;
      if (result) drawScatter(root);
    });
    dr.querySelectorAll('[data-t]').forEach((b) => b.addEventListener('click', () => openStock(root, b.dataset.t)));
    plot($('#dr-price', dr), [
      { x: d.price.dates, y: d.price.market, name: 'Market (equal weight)', line: { color: '#9aa5b4', width: 1.5 }, hovertemplate: '%{y:.2f}<extra>market</extra>' },
      { x: d.price.dates, y: d.price.value, name: d.ticker, line: { color: d.color, width: 2.4 }, hovertemplate: '%{y:.2f}<extra>' + d.ticker + '</extra>' },
    ], { yaxis: { title: { text: 'growth of $1' } }, margin: { l: 46, r: 8, t: 30, b: 28 }, hovermode: 'x unified' });
  }
})();

// ================================================================ view: hmm
const VIEW_hmm = (() => {
  const state = { n_states: 3, start: 1871, end: 2026 };
  let R = null;
  let tmIdx = 0;
  let timer = null;

  return {
    mount(root) {
      root.innerHTML = `
        <div class="view-head">
          <div>
            <h2>Market regimes</h2>
            <p>A Gaussian Hidden Markov Model: a hidden market state follows a Markov chain and each month's S&amp;P 500 return is drawn from that state's normal distribution. Baum-Welch learns the parameters; our own Viterbi and forward algorithm decode them.</p>
          </div>
          <span class="rubric">Rubric: implementation of HMM, 4 marks</span>
        </div>

        <div class="controls">
          <div class="field"><span>Hidden states</span>
            <div class="seg" id="hm-n"><button data-v="2">2</button><button data-v="3">3</button><button data-v="4">4</button><button data-v="5">5</button></div>
          </div>
          <label class="field"><span>Train from</span><select id="hm-start">${[1871, 1900, 1926, 1950, 1970, 1980].map((y) => `<option>${y}</option>`).join('')}</select></label>
          <label class="field"><span>Train to</span><select id="hm-end">${[2026, 2019, 2007, 1999].map((y) => `<option>${y}</option>`).join('')}</select></label>
          <span class="spacer"></span>
          <span class="compute" id="hm-chip"></span>
        </div>

        <div class="panel" id="hm-main">
          <div class="panel-head"><h3 id="hm-title">Decoded regimes</h3><div class="chips" id="hm-legend"></div></div>
          <p class="sub">Top: S&amp;P 500 on a log scale, each month coloured by its Viterbi state. Bottom: what the forward filter believed in real time, using no future data. Drag to zoom, double-click to reset.</p>
          <div class="plot" id="hm-chart" style="min-height:520px"></div>
        </div>

        <div class="grid g-main">
          <div class="panel">
            <div class="panel-head"><h3>Time machine</h3><div style="display:flex;gap:8px"><button class="btn" id="hm-play">Play from here</button>
              <select id="hm-jump" class="btn" aria-label="Jump to an event"><option value="">Jump to an event</option></select></div></div>
            <input type="range" id="hm-slider" min="0" max="100" value="0" style="width:100%;accent-color:var(--accent)" aria-label="Month">
            <div class="grid g2" style="align-items:center">
              <div style="display:flex;flex-direction:column;gap:6px">
                <span class="note" id="tm-date"></span>
                <span style="font:700 34px/1 var(--display)" id="tm-price"></span>
                <span class="note" id="tm-ret"></span>
                <span id="tm-state"></span>
              </div>
              <div class="bars" id="tm-bars" style="display:flex;flex-direction:column;gap:10px"></div>
            </div>
          </div>
          <div class="panel">
            <h3>Fit and verification</h3>
            <div id="hm-verify"></div>
          </div>
        </div>

        <div class="panel">
          <h3>What each state means</h3>
          <div class="scroll-x"><table id="hm-table"></table></div>
        </div>

        <div class="grid g2">
          <div class="panel">
            <h3>Transition matrix</h3>
            <p class="sub">Probability of moving from the row state this month to the column state next month. Arrow width in the diagram is drawn to scale.</p>
            <div class="grid g2" style="align-items:center"><div class="plot short" id="hm-A"></div><svg class="state-diagram" id="hm-diagram" viewBox="18 0 284 256" role="img" aria-label="State transition diagram"></svg></div>
          </div>
          <div class="panel">
            <h3>Emission distributions</h3>
            <p class="sub">Each state's normal distribution of monthly returns, weighted by how often it occurs, over the real histogram.</p>
            <div class="plot" id="hm-em"></div>
          </div>
        </div>

        <div class="grid g2">
          <div class="panel">
            <h3>Worst crisis episodes found</h3>
            <p class="sub">Unbroken runs of the most volatile state, ranked by total return. No dates were given to the model.</p>
            <div class="scroll-x"><table id="hm-eps"></table></div>
          </div>
          <div class="panel" id="hm-sel-panel">
            <div class="panel-head"><h3>How many states?</h3><span class="compute" id="hm-sel-chip"></span></div>
            <p class="sub" id="hm-sel-reason"></p>
            <div class="plot" id="hm-sel"></div>
          </div>
        </div>`;

      segmented($('#hm-n', root), state.n_states, (v) => { state.n_states = +v; fit(root); });
      $('#hm-start', root).addEventListener('change', (e) => { state.start = +e.target.value; fit(root); });
      $('#hm-end', root).addEventListener('change', (e) => { state.end = +e.target.value; fit(root); });
      const sl = $('#hm-slider', root);
      sl.addEventListener('input', () => { tmIdx = +sl.value; tm(root); });
      $('#hm-play', root).addEventListener('click', () => play(root));
      $('#hm-jump', root).addEventListener('change', (e) => {
        const i = R.series.dates.indexOf(e.target.value);
        if (i >= 0) { tmIdx = i; sl.value = i; tm(root); }
        e.target.value = '';
      });
      fit(root);
      loadSelect(root);
    },
  };

  async function fit(root) {
    stop(root);
    const msg = state.n_states >= 4 ? `Fitting a ${state.n_states}-state HMM (about 10 s the first time)…` : 'Fitting the HMM with Baum-Welch…';
    const call = await withBusy($('#hm-main', root), msg, () => api('hmm', state));
    R = call.data;
    computeChip($('#hm-chip', root), call);
    $('#hm-title', root).textContent = `${R.n_states} regimes decoded, ${R.window[0]}–${R.window[1]}`;
    $('#hm-legend', root).innerHTML = R.names.map((n, i) => `<span class="chip" style="cursor:default"><i class="dot" style="background:${R.colors[i]};margin:0"></i>${esc(n)}</span>`).join('');
    drawMain(root);
    drawTable(root);
    drawA(root);
    drawEm(root);
    drawEps(root);
    drawVerify(root);
    const sl = $('#hm-slider', root);
    sl.max = R.series.dates.length - 1;
    const crash = R.series.dates.indexOf('2008-10');
    tmIdx = crash >= 0 ? crash : R.series.dates.length - 1;
    sl.value = tmIdx;
    $('#hm-jump', root).innerHTML = '<option value="">Jump to an event</option>' + R.events.map((e) => `<option value="${e.date}">${esc(e.label)} (${e.date.slice(0, 4)})</option>`).join('') +
      `<option value="${R.series.dates[R.series.dates.length - 1]}">Latest month</option>`;
    tm(root);
  }

  function drawMain(root) {
    const s = R.series;
    const traces = [{ x: s.dates, y: s.price, type: 'scatter', mode: 'lines', line: { color: '#9aa5b4', width: 1 }, hoverinfo: 'skip', showlegend: false }];
    R.names.forEach((n, k) => {
      const idx = s.state.map((v, i) => (v === k ? i : -1)).filter((i) => i >= 0);
      traces.push({ type: 'scatter', mode: 'markers', name: n, x: idx.map((i) => s.dates[i]), y: idx.map((i) => s.price[i]),
        marker: { color: R.colors[k], size: 4.5 }, customdata: idx.map((i) => (s.ret[i] * 100).toFixed(2)),
        hovertemplate: `%{x|%b %Y}<br>S&P %{y:,.2f}<br>return %{customdata}%<br><b>${n}</b><extra></extra>` });
    });
    R.names.forEach((n, k) => traces.push({ x: s.dates, y: s.probs[k], type: 'scatter', mode: 'lines', stackgroup: 'p', name: 'P(' + n + ')', showlegend: false,
      line: { width: 0, color: R.colors[k] }, fillcolor: hexA(R.colors[k], 0.85), xaxis: 'x', yaxis: 'y2', hovertemplate: `P(${n}) %{y:.0%}<extra></extra>` }));
    const narrow = $('#hm-chart', root).clientWidth < 640;
    const ann = narrow ? [] : R.events.filter((e) => e.date < '2022').map((e, n) => {
      const i = s.dates.indexOf(e.date);
      return i < 0 ? null : { x: e.date, y: Math.log10(s.price[i]), text: e.label, showarrow: true, arrowhead: 0, ax: 0, ay: n % 2 ? -52 : -26, font: { size: 11, color: INK.ink2 }, arrowcolor: INK.ink3 };
    }).filter(Boolean);
    plot($('#hm-chart', root), traces, {
      xaxis: { type: 'date', anchor: 'y2' },
      yaxis: { type: 'log', domain: [0.34, 1], title: { text: 'S&P 500 (log)' } },
      yaxis2: { domain: [0, 0.26], range: [0, 1], tickformat: '.0%', title: { text: 'belief' } },
      annotations: ann, legend: { orientation: 'h', y: 1.06 }, margin: { l: 64, r: 16, t: 24, b: 36 },
      shapes: [cursorShape(s.dates[tmIdx])],
    });
  }

  function cursorShape(x) { return { type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1, line: { color: '#132033', width: 1, dash: 'dot' } }; }

  function tm(root) {
    const s = R.series;
    const i = tmIdx;
    const k = s.state[i];
    const dt = new Date(s.dates[i] + '-01T00:00').toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
    $('#tm-date', root).textContent = dt;
    $('#tm-price', root).textContent = s.price[i].toLocaleString(undefined, { maximumFractionDigits: 2 });
    $('#tm-ret', root).textContent = `S&P 500 level · monthly return ${spct(s.ret[i], 2)}`;
    $('#tm-state', root).innerHTML = `<span class="chip" style="cursor:default;font-size:15px"><i class="dot" style="background:${R.colors[k]};margin:0;width:12px;height:12px"></i>Viterbi says <b>${esc(R.names[k])}</b></span>`;
    $('#tm-bars', root).innerHTML = '<span class="note">Real-time belief that month</span>' + R.names.map((n, j) =>
      `<div class="pct-row" style="grid-template-columns:96px 1fr 52px"><span>${esc(n)}</span><span class="bar"><span style="width:${(s.probs[j][i] * 100).toFixed(1)}%;background:${R.colors[j]}"></span></span><span>${pct(s.probs[j][i])}</span></div>`).join('');
    Plotly.relayout($('#hm-chart', root), { shapes: [cursorShape(s.dates[i])] });
  }

  function play(root) {
    if (timer) return stop(root);
    const sl = $('#hm-slider', root);
    if (tmIdx >= R.series.dates.length - 1) tmIdx = 0;
    $('#hm-play', root).textContent = 'Pause';
    timer = setInterval(() => {
      tmIdx = Math.min(R.series.dates.length - 1, tmIdx + 1);
      sl.value = tmIdx;
      tm(root);
      if (tmIdx >= R.series.dates.length - 1) stop(root);
    }, 120);
  }
  function stop(root) { clearInterval(timer); timer = null; const b = $('#hm-play', root); if (b) b.textContent = 'Play from here'; }

  function drawTable(root) {
    $('#hm-table', root).innerHTML = '<tr><th>State</th><th>Return / yr</th><th>Volatility / yr</th><th>Share of months</th><th>Stays next month</th><th>Average length</th></tr>' +
      R.states.map((s) => `<tr><td><i class="dot" style="background:${s.color}"></i><b>${esc(s.name)}</b></td><td>${spct(s.ann_return)}</td><td>${pct(s.ann_vol, 1)}</td><td>${pct(s.share)}</td><td>${pct(s.persistence)}</td><td>${num(s.expected_months, 1)} months</td></tr>`).join('');
  }

  function drawA(root) {
    const A = R.transmat;
    plot($('#hm-A', root), [{ type: 'heatmap', z: A, x: R.names, y: R.names, text: A.map((r) => r.map((v) => v.toFixed(2))), texttemplate: '%{text}', textfont: { size: 13 },
      colorscale: [[0, '#f5f7fa'], [1, '#132033']], zmin: 0, zmax: 1, showscale: false, xgap: 2, ygap: 2,
      hovertemplate: '%{y} → %{x}: %{z:.3f}<extra></extra>' }], {
      yaxis: { autorange: 'reversed', title: { text: 'this month' }, gridcolor: 'rgba(0,0,0,0)' }, xaxis: { title: { text: 'next month' }, gridcolor: 'rgba(0,0,0,0)' }, margin: { l: 90, r: 4, t: 4, b: 50 } });
    // diagram
    const n = R.names.length;
    const cx = 160, cy = 132, rad = n === 2 ? 90 : 92;
    const P = R.names.map((_, i) => { const a = -Math.PI / 2 + (2 * Math.PI * i) / n + (n === 2 ? Math.PI / 2 : 0); return [cx + rad * Math.cos(a), cy + rad * Math.sin(a)]; });
    let g = '<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L8,4L0,8z" fill="#7a8798"/></marker></defs>';
    const R0 = n > 3 ? 27 : 34;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      if (i === j || A[i][j] < 0.01) continue;
      const [x1, y1] = P[i], [x2, y2] = P[j];
      const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy), nx = -dy / L * 6, ny = dx / L * 6;
      const sx = x1 + dx / L * (R0 + 2) + nx, sy = y1 + dy / L * (R0 + 2) + ny, ex = x2 - dx / L * (R0 + 4) + nx, ey = y2 - dy / L * (R0 + 4) + ny;
      g += `<line x1="${sx}" y1="${sy}" x2="${ex}" y2="${ey}" stroke="#7a8798" stroke-width="${Math.max(1, A[i][j] * 10)}" marker-end="url(#ah)"/>`;
      g += `<text x="${(sx + ex) / 2 + nx * 2.2}" y="${(sy + ey) / 2 + ny * 2.2 + 4}" font-size="12" fill="#4a5a6e" text-anchor="middle">${A[i][j].toFixed(2)}</text>`;
    }
    R.names.forEach((nm, i) => {
      g += `<circle cx="${P[i][0]}" cy="${P[i][1]}" r="${R0}" fill="${R.colors[i]}"/>`;
      g += `<text x="${P[i][0]}" y="${P[i][1] - 1}" font-size="${n > 3 ? 11 : 13}" font-weight="600" fill="#132033" text-anchor="middle">${esc(nm)}</text>`;
      g += `<text x="${P[i][0]}" y="${P[i][1] + 12}" font-size="11" fill="#132033" text-anchor="middle">stay ${A[i][i].toFixed(2)}</text>`;
    });
    $('#hm-diagram', root).innerHTML = g;
  }

  function drawEm(root) {
    const e = R.emission;
    const tr = [{ type: 'bar', x: e.hist_x.map((v) => v * 100), y: e.hist_y.map((v) => v / 100), marker: { color: '#dde3ea' }, name: 'Actual months', hoverinfo: 'skip' }];
    R.names.forEach((n, i) => tr.push({ x: e.x.map((v) => v * 100), y: e.density[i].map((v) => v / 100), name: n, mode: 'lines', line: { color: R.colors[i], width: 2.6 }, hovertemplate: `${n}: %{x:.1f}%<extra></extra>` }));
    plot($('#hm-em', root), tr, { xaxis: { title: { text: 'monthly return (%)' }, range: [-20, 15] }, yaxis: { title: { text: 'density' }, showticklabels: false }, bargap: 0, margin: { l: 40, r: 8, t: 30, b: 42 } });
  }

  function drawEps(root) {
    $('#hm-eps', root).innerHTML = '<tr><th>From</th><th>To</th><th>Months</th><th>S&amp;P return</th></tr>' +
      R.episodes.slice(0, 8).map((e) => `<tr><td>${e.start}</td><td>${e.end}</td><td>${e.months}</td><td>${spct(e.return)}</td></tr>`).join('');
  }

  function drawVerify(root) {
    const v = R.verification, f = R.fit;
    $('#hm-verify', root).innerHTML = `
      <div class="kpis">
        <div class="kpi"><span class="v">${num(f.loglik, 1)}</span><span class="l">log-likelihood</span></div>
        <div class="kpi"><span class="v">${num(f.bic, 1)}</span><span class="l">BIC (${f.n_params} parameters, ${f.n_obs} months)</span></div>
      </div>
      <div class="verify" style="flex-direction:column">
        <span>Our NumPy forward algorithm: <b>${num(v.loglik_scratch, 4)}</b></span>
        <span>hmmlearn library: <b>${num(v.loglik_hmmlearn, 4)}</b> ${Math.abs(v.loglik_scratch - v.loglik_hmmlearn) < 1e-6 ? '<span class="ok-tag">match</span>' : ''}</span>
        <span>Our Viterbi path agrees with hmmlearn on <b>${pct(v.viterbi_agreement, 1)}</b> of months</span>
      </div>`;
  }

  async function loadSelect(root) {
    const call = await withBusy($('#hm-sel-panel', root), 'Fitting HMMs with 1–5 states…', () => api('hmm/select'));
    computeChip($('#hm-sel-chip', root), call);
    const d = call.data;
    $('#hm-sel-reason', root).textContent = d.reason;
    plot($('#hm-sel', root), [
      { x: d.rows.map((r) => r.states), y: d.rows.map((r) => r.bic), name: 'BIC', mode: 'lines+markers', line: { color: '#2b63d9', width: 2.4 }, marker: { size: 8 } },
      { x: d.rows.map((r) => r.states), y: d.rows.map((r) => r.aic), name: 'AIC', mode: 'lines+markers', line: { color: '#ff6b3d', width: 2 }, marker: { size: 7 } },
    ], { xaxis: { dtick: 1, title: { text: 'hidden states' } }, yaxis: { title: { text: 'lower is better' } },
      shapes: [{ type: 'line', x0: d.chosen, x1: d.chosen, yref: 'paper', y0: 0, y1: 1, line: { color: '#132033', dash: 'dot' } }], margin: { l: 60, r: 8, t: 30, b: 42 } });
  }
})();

// ================================================================ view: lab
const VIEW_lab = (() => {
  const PAL = ['#3d8bfd', '#ff7a45', '#20c997', '#f5b301', '#a78bfa', '#f06595', '#63e6be', '#ffa94d'];
  const km = { k: 4, dataset: 'stocks', seed: 1, data: null, f: 0, timer: null };
  const vt = { start: '2007-06', length: 18, data: null, t: 0, back: -1, timer: null };

  return {
    mount(root) {
      root.innerHTML = `
        <div class="view-head">
          <div>
            <h2>Algorithm lab</h2>
            <p>Both algorithms below run on the server in our own NumPy code. The server returns every intermediate step, and the page replays them so you can watch each one work.</p>
          </div>
          <span class="rubric">Innovation: from-scratch algorithms, 4 marks</span>
        </div>

        <div class="panel" id="km-panel">
          <div class="panel-head"><h3>K-Means, one step at a time</h3><span class="compute" id="km-chip"></span></div>
          <p class="sub">Lloyd's algorithm repeats two steps: assign each point to its nearest centroid, then move each centroid to the mean of its points. It stops when no point changes cluster.</p>
          <div class="controls" style="padding:12px 0;border:0">
            <div class="field"><span>Data</span><div class="seg" id="km-ds"><button data-v="stocks">477 real stocks</button><button data-v="blobs">Four blobs</button><button data-v="moons">Two moons</button></div></div>
            <label class="field"><span>k = <output id="km-k-o"></output></span><input type="range" id="km-k" min="2" max="8" value="4"></label>
            <span class="spacer"></span>
            <button class="btn" id="km-new">New random start</button>
            <button class="btn" id="km-reset">Reset</button>
            <button class="btn" id="km-step">Step</button>
            <button class="btn primary" id="km-play">Play</button>
          </div>
          <div class="grid g-main">
            <canvas class="lab-canvas" id="km-cv" aria-label="K-Means animation"></canvas>
            <div style="display:flex;flex-direction:column;gap:14px">
              <div class="kpis" style="grid-template-columns:1fr 1fr">
                <div class="kpi"><span class="v" id="km-it">0</span><span class="l">iteration</span></div>
                <div class="kpi"><span class="v" id="km-in">—</span><span class="l">inertia (within-cluster spread)</span></div>
              </div>
              <p class="trellis-note" id="km-msg"></p>
              <canvas class="lab-canvas" id="km-hist" aria-label="Inertia per iteration"></canvas>
              <p class="note" id="km-space"></p>
            </div>
          </div>
        </div>

        <div class="panel" id="vt-panel">
          <div class="panel-head"><h3>Viterbi, month by month</h3><span class="compute" id="vt-chip"></span></div>
          <p class="sub">Rows are hidden states, columns are months. Circle size is δ, the probability of the best path that ends in that state. Grey lines are back-pointers ψ. After the last month, the algorithm walks the back-pointers to read off the single most likely sequence of regimes.</p>
          <div class="controls" style="padding:12px 0;border:0">
            <label class="field"><span>Episode</span><select id="vt-ep">
              <option value="2007-06">2007–09 financial crisis</option><option value="1929-06">1929–31 crash</option>
              <option value="2019-09">2019–21 COVID shock</option><option value="1999-10">1999–2002 dot-com bust</option>
              <option value="1973-01">1973–74 oil shock</option><option value="2021-10">2021–23 rate shock</option></select></label>
            <label class="field"><span>Months = <output id="vt-len-o"></output></span><input type="range" id="vt-len" min="8" max="30" value="18"></label>
            <span class="spacer"></span>
            <button class="btn" id="vt-reset">Reset</button>
            <button class="btn" id="vt-step">Step</button>
            <button class="btn primary" id="vt-play">Run all</button>
          </div>
          <canvas class="lab-canvas" id="vt-cv" aria-label="Viterbi trellis"></canvas>
          <p class="trellis-note" id="vt-note"></p>
        </div>`;

      segmented($('#km-ds', root), km.dataset, (v) => { km.dataset = v; km.k = v === 'moons' ? 2 : km.k; $('#km-k', root).value = km.k; $('#km-k-o', root).textContent = km.k; loadKM(root); });
      const k = $('#km-k', root);
      rangeField(k, $('#km-k-o', root));
      k.addEventListener('change', () => { km.k = +k.value; loadKM(root); });
      $('#km-new', root).addEventListener('click', () => { km.seed = Math.floor(Math.random() * 999); loadKM(root); });
      $('#km-reset', root).addEventListener('click', () => { stopKM(root); km.f = 0; drawKM(root); });
      $('#km-step', root).addEventListener('click', () => { stopKM(root); stepKM(root); });
      $('#km-play', root).addEventListener('click', () => playKM(root));

      $('#vt-ep', root).addEventListener('change', (e) => { vt.start = e.target.value; loadVT(root); });
      const len = $('#vt-len', root);
      rangeField(len, $('#vt-len-o', root));
      len.addEventListener('change', () => { vt.length = +len.value; loadVT(root); });
      $('#vt-reset', root).addEventListener('click', () => { stopVT(root); vt.t = 0; vt.back = -1; drawVT(root); });
      $('#vt-step', root).addEventListener('click', () => { stopVT(root); stepVT(root); });
      $('#vt-play', root).addEventListener('click', () => playVT(root));

      loadKM(root);
      loadVT(root);
      window.addEventListener('resize', () => { if (!root.hidden) { km.data && drawKM(root); vt.data && drawVT(root); } });
    },
    onShow() {
      const root = document.getElementById('view-lab');
      km.data && drawKM(root);
      vt.data && drawVT(root);
    },
  };

  // ---------------------------------------------------------------- K-Means
  async function loadKM(root) {
    stopKM(root);
    const call = await withBusy($('#km-panel', root), 'Running K-Means on the server…', () => api('lab/kmeans', { k: km.k, dataset: km.dataset, seed: km.seed }));
    km.data = call.data;
    km.f = 0;
    computeChip($('#km-chip', root), call);
    $('#km-space', root).textContent = km.dataset === 'stocks'
      ? 'On real stocks the algorithm works in all six feature dimensions. The picture shows the first two principal components, so a few points can look closer to another centroid than they really are.'
      : km.dataset === 'moons' ? 'Two moons is the classic case K-Means gets wrong: it can only draw straight boundaries between round clusters.' : 'Four well-separated blobs: K-Means finds them in a few iterations.';
    drawKM(root);
  }

  function stepKM(root) {
    if (km.f < km.data.frames.length - 1) { km.f++; drawKM(root); return true; }
    return false;
  }
  function playKM(root) {
    if (km.timer) return stopKM(root);
    if (km.f >= km.data.frames.length - 1) km.f = 0;
    $('#km-play', root).textContent = 'Pause';
    km.timer = setInterval(() => { if (!stepKM(root)) stopKM(root); }, 520);
  }
  function stopKM(root) { clearInterval(km.timer); km.timer = null; const b = $('#km-play', root); if (b) b.textContent = 'Play'; }

  function drawKM(root) {
    const cv = $('#km-cv', root);
    if (!cv.clientWidth) return;
    const { c, w, h } = resizeCanvas(cv, Math.min(460, Math.max(320, cv.clientWidth * 0.62)));
    const d = km.data, fr = d.frames[km.f];
    const xs = d.points.map((p) => p[0]), ys = d.points.map((p) => p[1]);
    const allC = d.frames.flatMap((f) => f.centroids);
    const x0 = Math.min(...xs, ...allC.map((q) => q[0])) - 0.4, x1 = Math.max(...xs, ...allC.map((q) => q[0])) + 0.4;
    const y0 = Math.min(...ys, ...allC.map((q) => q[1])) - 0.4, y1 = Math.max(...ys, ...allC.map((q) => q[1])) + 0.4;
    const X = (v) => 14 + (w - 28) * (v - x0) / (x1 - x0), Y = (v) => 14 + (h - 28) * (1 - (v - y0) / (y1 - y0));

    if (d.dataset !== 'stocks' && fr.phase !== 'init') {
      const st = 8;
      for (let px = 0; px < w; px += st) for (let py = 0; py < h; py += st) {
        const vx = x0 + (px - 14) / (w - 28) * (x1 - x0), vy = y0 + (1 - (py - 14) / (h - 28)) * (y1 - y0);
        let b = 0, bd = 1e9;
        fr.centroids.forEach((q, j) => { const dd = (vx - q[0]) ** 2 + (vy - q[1]) ** 2; if (dd < bd) { bd = dd; b = j; } });
        c.fillStyle = hexA(PAL[b % 8], 0.08); c.fillRect(px, py, st, st);
      }
    }
    d.points.forEach((p, i) => {
      const l = fr.labels[i];
      c.beginPath(); c.arc(X(p[0]), Y(p[1]), 3.6, 0, 7);
      c.fillStyle = l < 0 ? '#aab4c2' : PAL[l % 8]; c.fill();
    });
    // centroid trails
    const upto = d.frames.slice(0, km.f + 1);
    fr.centroids.forEach((_, j) => {
      c.strokeStyle = 'rgba(19,32,51,0.45)'; c.lineWidth = 1.2; c.beginPath();
      upto.forEach((f, t) => { const q = f.centroids[j]; t ? c.lineTo(X(q[0]), Y(q[1])) : c.moveTo(X(q[0]), Y(q[1])); });
      c.stroke();
    });
    fr.centroids.forEach((q, j) => {
      const x = X(q[0]), y = Y(q[1]);
      c.fillStyle = PAL[j % 8]; c.strokeStyle = '#132033'; c.lineWidth = 2.2;
      c.beginPath(); c.moveTo(x, y - 11); c.lineTo(x + 11, y); c.lineTo(x, y + 11); c.lineTo(x - 11, y); c.closePath(); c.fill(); c.stroke();
    });

    const iter = d.frames.slice(0, km.f + 1).filter((f) => f.phase === 'update').length;
    $('#km-it', root).textContent = iter;
    $('#km-in', root).textContent = fr.inertia == null ? '—' : fr.inertia.toFixed(1);
    const last = km.f === d.frames.length - 1;
    $('#km-msg', root).innerHTML = fr.phase === 'init' ? 'Start: k centroids placed with k-means++ seeding (spread apart on purpose). Press Step.'
      : fr.phase === 'assign' ? `<b>Assign step.</b> Every point joins its nearest centroid. ${fr.moved} points changed cluster.`
      : last ? `<b>Converged</b> after ${iter} iterations: no point changed cluster and the centroids stopped moving.`
      : '<b>Update step.</b> Each centroid moves to the mean of its points (trails show the path so far).';

    const hc = $('#km-hist', root);
    const H = resizeCanvas(hc, 90);
    const vals = d.frames.filter((f, i) => i <= km.f && f.phase === 'assign').map((f) => f.inertia);
    const all = d.frames.filter((f) => f.phase === 'assign').map((f) => f.inertia);
    if (all.length > 1) {
      const mx = Math.max(...all), mn = Math.min(...all);
      const xx = (i) => 8 + (H.w - 16) * i / (all.length - 1), yy = (v) => 10 + (H.h - 20) * (mx === mn ? 0.5 : 1 - (v - mn) / (mx - mn));
      H.c.strokeStyle = '#dde3ea'; H.c.lineWidth = 1; H.c.beginPath(); all.forEach((v, i) => (i ? H.c.lineTo(xx(i), yy(v)) : H.c.moveTo(xx(i), yy(v)))); H.c.stroke();
      H.c.strokeStyle = '#2b63d9'; H.c.lineWidth = 2.2; H.c.beginPath(); vals.forEach((v, i) => (i ? H.c.lineTo(xx(i), yy(v)) : H.c.moveTo(xx(i), yy(v)))); H.c.stroke();
      if (vals.length) { H.c.fillStyle = '#2b63d9'; H.c.beginPath(); H.c.arc(xx(vals.length - 1), yy(vals[vals.length - 1]), 4, 0, 7); H.c.fill(); }
    }
  }

  // ---------------------------------------------------------------- Viterbi
  async function loadVT(root) {
    stopVT(root);
    const call = await withBusy($('#vt-panel', root), 'Decoding on the server…', () => api('hmm/viterbi', { start: vt.start, length: vt.length }));
    vt.data = call.data;
    vt.t = 0; vt.back = -1;
    computeChip($('#vt-chip', root), call);
    drawVT(root);
  }

  function stepVT(root) {
    const T = vt.data.dates.length;
    if (vt.t < T) vt.t++;
    else if (vt.back < 0) vt.back = T - 1;
    else if (vt.back > 0) vt.back--;
    else return false;
    drawVT(root);
    return true;
  }
  function playVT(root) {
    if (vt.timer) return stopVT(root);
    if (vt.back === 0) { vt.t = 0; vt.back = -1; }
    $('#vt-play', root).textContent = 'Pause';
    vt.timer = setInterval(() => { if (!stepVT(root)) stopVT(root); }, 260);
  }
  function stopVT(root) { clearInterval(vt.timer); vt.timer = null; const b = $('#vt-play', root); if (b) b.textContent = 'Run all'; }

  function drawVT(root) {
    const cv = $('#vt-cv', root);
    if (!cv.clientWidth) return;
    const d = vt.data, T = d.dates.length, N = d.names.length;
    const { c, w, h } = resizeCanvas(cv, 400);
    const L = 104, Rm = 14, top = 92, bot = 34;
    const cw = (w - L - Rm) / T;
    const rows = d.names.map((_, k) => top + 24 + k * ((h - top - bot - 48) / Math.max(1, N - 1)));
    c.font = '12px "IBM Plex Sans", system-ui, sans-serif';
    // returns strip
    const rmax = Math.max(0.05, ...d.returns.map(Math.abs));
    c.fillStyle = '#7a8798'; c.fillText('monthly return', 8, 46);
    c.strokeStyle = '#dde3ea'; c.beginPath(); c.moveTo(L, 46); c.lineTo(w - Rm, 46); c.stroke();
    d.returns.forEach((r, t) => {
      const x = L + cw * (t + 0.5), bh = r / rmax * 34;
      c.fillStyle = r >= 0 ? '#20c997' : '#ff6b3d';
      c.fillRect(x - cw * 0.28, r >= 0 ? 46 - bh : 46, cw * 0.56, Math.abs(bh));
      if (T <= 20 || t % 2 === 0) { c.fillStyle = '#7a8798'; c.textAlign = 'center'; c.fillText(d.dates[t].slice(2).replace('-', '/'), x, h - 10); c.textAlign = 'left'; }
    });
    d.names.forEach((n, k) => { c.fillStyle = '#132033'; c.font = '600 13px "IBM Plex Sans", system-ui, sans-serif'; c.fillText(n, 8, rows[k] + 4); });
    c.font = '12px "IBM Plex Sans", system-ui, sans-serif';
    // back-pointers
    for (let t = 1; t < vt.t; t++) for (let j = 0; j < N; j++) {
      const i = d.psi[t][j];
      c.strokeStyle = 'rgba(122,135,152,0.55)'; c.lineWidth = 1;
      c.beginPath(); c.moveTo(L + cw * (t - 0.5), rows[i]); c.lineTo(L + cw * (t + 0.5), rows[j]); c.stroke();
    }
    // backtracked path
    if (vt.back >= 0) {
      c.strokeStyle = '#132033'; c.lineWidth = 3.2; c.beginPath();
      for (let t = T - 1; t >= vt.back; t--) { const x = L + cw * (t + 0.5), y = rows[d.path[t]]; t === T - 1 ? c.moveTo(x, y) : c.lineTo(x, y); }
      c.stroke();
    }
    for (let t = 0; t < T; t++) for (let k = 0; k < N; k++) {
      const x = L + cw * (t + 0.5), y = rows[k];
      const done = t < vt.t;
      const onPath = vt.back >= 0 && t >= vt.back && d.path[t] === k;
      const r = done ? 4 + Math.sqrt(d.delta[t][k]) * Math.min(15, cw * 0.42) : 3;
      c.globalAlpha = done && vt.back >= 0 && t >= vt.back && !onPath ? 0.28 : 1;
      c.beginPath(); c.arc(x, y, r, 0, 7);
      c.fillStyle = done ? d.colors[k] : '#dde3ea'; c.fill();
      c.globalAlpha = 1;
      if (onPath) { c.strokeStyle = '#132033'; c.lineWidth = 2.4; c.stroke(); }
    }
    const note = $('#vt-note', root);
    if (vt.t === 0) note.innerHTML = 'Press Step to compute δ₁(j) = π<sub>j</sub> · b<sub>j</sub>(r₁) for the first month: how likely each state is before any history.';
    else if (vt.back < 0) {
      const t = vt.t - 1;
      const best = d.delta[t].indexOf(Math.max(...d.delta[t]));
      note.innerHTML = `Month ${t + 1} (${d.dates[t]}), return ${spct(d.returns[t], 2)}. δ<sub>t</sub>(j) = max<sub>i</sub> [δ<sub>t−1</sub>(i) · a<sub>ij</sub>] · b<sub>j</sub>(r<sub>t</sub>). Best state so far: <b style="color:${d.colors[best]}">${esc(d.names[best])}</b>.` +
        (vt.t === T ? ' Forward pass finished. Step again to backtrack.' : '');
    } else if (vt.back > 0) note.innerHTML = `Backtracking through ψ: month ${vt.back + 1} (${d.dates[vt.back]}) is <b style="color:${d.colors[d.path[vt.back]]}">${esc(d.names[d.path[vt.back]])}</b>.`;
    else {
      const runs = [];
      d.path.forEach((k) => { if (runs.length && runs[runs.length - 1][0] === k) runs[runs.length - 1][1]++; else runs.push([k, 1]); });
      note.innerHTML = '<b>Decoded path:</b> ' + runs.map(([k, n]) => `<span style="color:${d.colors[k]};font-weight:600">${esc(d.names[k])}</span> for ${n} month${n > 1 ? 's' : ''}`).join(', then ') + '.';
    }
  }
})();

// ================================================================ view: compare
const VIEW_compare = (() => {
  let flick = null;
  let win = '2005';

  return {
    mount(root) {
      root.innerHTML = `
        <div class="view-head">
          <div>
            <h2>Comparison</h2>
            <p>Three questions: which clustering algorithm fits the stocks best, does the HMM beat plain clustering for regimes, and do the regimes agree with a completely independent signal?</p>
          </div>
          <span class="rubric">Rubric: comparative performance analysis, 2 marks</span>
        </div>

        <div class="panel" id="cp-cl">
          <div class="panel-head"><h3>Clustering algorithms, head to head</h3><span class="compute" id="cp-cl-chip"></span></div>
          <p class="sub">Same 477 stocks, k = 4 (DBSCAN chooses its own count). Stability re-runs each algorithm on bootstrap resamples and measures how much the labels agree (adjusted Rand index).</p>
          <div class="grid g-main">
            <div class="scroll-x"><table id="cp-cl-table"></table></div>
            <div class="plot" id="cp-cl-plot"></div>
          </div>
          <p class="note" id="cp-cl-verdict" style="margin:0"></p>
        </div>

        <div class="panel" id="cp-rg">
          <div class="panel-head"><h3>Why an HMM? Memory beats memory-less clustering</h3><span class="compute" id="cp-rg-chip"></span></div>
          <p class="sub">K-Means and a Gaussian mixture also find three groups in the monthly returns, but they judge every month on its own. The HMM's transition matrix remembers last month, so regimes persist instead of flickering.</p>
          <div class="controls" style="padding:4px 0;border:0">
            <div class="field"><span>Window</span><div class="seg" id="cp-win"><button data-v="2005">2005 – today</button><button data-v="1925">1925 – 1945</button><button data-v="1871">All 155 years</button></div></div>
          </div>
          <div class="plot" id="cp-flicker" style="min-height:230px"></div>
          <div class="scroll-x"><table id="cp-rg-table"></table></div>
        </div>

        <div class="panel" id="cp-vx">
          <div class="panel-head"><h3>Independent check: an HMM on the VIX fear index</h3><span class="compute" id="cp-vx-chip"></span></div>
          <p class="sub">A separate 3-state HMM trained only on daily VIX levels (1990 – today). If the S&amp;P regimes are real, the VIX model should be in its Fear state when the S&amp;P model says Crisis.</p>
          <div class="grid g-main">
            <div class="plot" id="cp-vix"></div>
            <div style="display:flex;flex-direction:column;gap:12px">
              <h4 style="margin:0">Share of days in VIX Fear, by S&amp;P regime</h4>
              <div class="plot short" id="cp-agree"></div>
              <p class="note" id="cp-vx-levels" style="margin:0"></p>
            </div>
          </div>
        </div>`;
      segmented($('#cp-win', root), win, (v) => { win = v; drawFlicker(root); });
      loadClusters(root);
      loadRegimes(root);
      loadVix(root);
    },
  };

  async function loadClusters(root) {
    const call = await withBusy($('#cp-cl', root), 'Running all four algorithms with 10 bootstrap refits each…', () => api('cluster/compare'));
    computeChip($('#cp-cl-chip', root), call);
    const d = call.data, rows = d.rows;
    const cols = [['silhouette', 'Silhouette ↑', 'max', 3], ['davies_bouldin', 'Davies-Bouldin ↓', 'min', 2], ['calinski_harabasz', 'Calinski-H. ↑', 'max', 0], ['stability', 'Stability ↑', 'max', 2], ['ari_vs_sector', 'ARI vs sector', null, 3], ['time_ms', 'Time (ms)', null, 0]];
    const best = {};
    cols.forEach(([k, , dir]) => { if (!dir) return; const v = rows.filter((r) => r.algo !== 'dbscan' && r[k] != null).map((r) => r[k]); best[k] = dir === 'max' ? Math.max(...v) : Math.min(...v); });
    $('#cp-cl-table', root).innerHTML = `<tr><th>Algorithm</th><th>Clusters</th><th>Outliers</th>${cols.map((c) => `<th>${c[1]}</th>`).join('')}</tr>` +
      rows.map((r) => `<tr><td><b>${esc(r.label)}</b></td><td>${r.clusters}</td><td>${r.noise}</td>${cols.map(([k, , , dp]) => `<td class="${best[k] === r[k] ? 'best' : ''}">${r[k] == null ? '—' : num(r[k], dp)}</td>`).join('')}</tr>`).join('');
    const sub = rows.filter((r) => r.algo !== 'dbscan');
    plot($('#cp-cl-plot', root), [
      { type: 'bar', name: 'Silhouette', x: sub.map((r) => r.label.split(' (')[0]), y: sub.map((r) => r.silhouette), marker: { color: '#2b63d9' }, text: sub.map((r) => r.silhouette.toFixed(2)), textposition: 'outside' },
      { type: 'bar', name: 'Stability', x: sub.map((r) => r.label.split(' (')[0]), y: sub.map((r) => r.stability), marker: { color: '#20c997' }, text: sub.map((r) => r.stability.toFixed(2)), textposition: 'outside' },
    ], { barmode: 'group', yaxis: { range: [0, 1], title: { text: 'higher is better' } }, margin: { l: 50, r: 8, t: 30, b: 36 } });
    $('#cp-cl-verdict', root).innerHTML = `<b>Verdict.</b> ${esc(d.verdict)}`;
  }

  async function loadRegimes(root) {
    const call = await withBusy($('#cp-rg', root), 'Fitting K-Means, GMM and the HMM on 1,868 months…', () => api('hmm/compare'));
    computeChip($('#cp-rg-chip', root), call);
    flick = call.data;
    const rows = flick.rows;
    const hmm = rows.find((r) => r.model === 'HMM');
    $('#cp-rg-table', root).innerHTML = '<tr><th>Model</th><th>Regime switches</th><th>Average regime length</th><th>Log-likelihood</th><th>BIC ↓</th></tr>' +
      rows.map((r) => `<tr><td><b>${r.model}</b></td><td class="${r === hmm ? 'best' : ''}">${r.switches}</td><td class="${r === hmm ? 'best' : ''}">${num(r.avg_months, 1)} months</td><td>${r.loglik == null ? '—' : num(r.loglik, 1)}</td><td class="${r === hmm ? 'best' : ''}">${r.bic == null ? '—' : num(r.bic, 1)}</td></tr>`).join('');
    drawFlicker(root);
  }

  function drawFlicker(root) {
    if (!flick) return;
    const from = { 2005: '2005-01', 1925: '1925-01', 1871: '1871-01' }[win];
    const to = win === '1925' ? '1945-12' : '9999';
    const idx = flick.dates.map((d, i) => (d >= from && d <= to ? i : -1)).filter((i) => i >= 0);
    const models = ['K-Means', 'GMM', 'HMM'];
    const z = models.map((m) => idx.map((i) => flick.labels[m][i]));
    const cs = [[0, flick.colors[0]], [0.333, flick.colors[0]], [0.334, flick.colors[1]], [0.666, flick.colors[1]], [0.667, flick.colors[2]], [1, flick.colors[2]]];
    plot($('#cp-flicker', root), [{ type: 'heatmap', z, x: idx.map((i) => flick.dates[i]), y: models, zmin: 0, zmax: 2, colorscale: cs, showscale: false, ygap: 6,
      customdata: z.map((r) => r.map((v) => flick.names[v])), hovertemplate: '%{y}, %{x}: <b>%{customdata}</b><extra></extra>' }],
    { yaxis: { autorange: 'reversed', gridcolor: 'rgba(0,0,0,0)' }, xaxis: { type: 'date', gridcolor: 'rgba(0,0,0,0)' }, margin: { l: 70, r: 8, t: 8, b: 30 },
      annotations: flick.names.map((n, i) => ({ xref: 'paper', yref: 'paper', x: i * 0.12, y: 1.12, xanchor: 'left', showarrow: false, text: `<span style="color:${flick.colors[i]}">■</span> ${n}`, font: { size: 12.5, color: INK.ink2 } })) });
  }

  async function loadVix(root) {
    const call = await withBusy($('#cp-vx', root), 'Fitting the VIX model…', () => api('vix'));
    computeChip($('#cp-vx-chip', root), call);
    const d = call.data, s = d.series;
    plot($('#cp-vix', root), [{ x: s.dates, y: s.vix, mode: 'lines', line: { color: '#c3ccd5', width: 1 }, hoverinfo: 'skip', showlegend: false },
      ...d.names.map((n, k) => {
        const idx = s.state.map((v, i) => (v === k ? i : -1)).filter((i) => i >= 0);
        return { type: 'scatter', mode: 'markers', name: n, x: idx.map((i) => s.dates[i]), y: idx.map((i) => s.vix[i]), marker: { color: d.colors[k], size: 4 }, hovertemplate: `%{x|%d %b %Y}: VIX %{y:.1f}<br><b>${n}</b><extra></extra>` };
      })], { yaxis: { title: { text: 'VIX' } }, xaxis: { type: 'date' }, margin: { l: 50, r: 8, t: 30, b: 30 } });
    const order = ['Bull', 'Correction', 'Crisis'].filter((k) => k in d.fear_share_by_sp_regime);
    const col = { Bull: '#20c997', Correction: '#3d8bfd', Crisis: '#ff6b3d' };
    plot($('#cp-agree', root), [{ type: 'bar', orientation: 'h', y: order, x: order.map((k) => d.fear_share_by_sp_regime[k]), marker: { color: order.map((k) => col[k]) },
      text: order.map((k) => pct(d.fear_share_by_sp_regime[k])), textposition: 'outside', hovertemplate: '%{y}: %{x:.0%}<extra></extra>' }],
    { xaxis: { range: [0, 1.15], tickformat: '.0%' }, yaxis: { autorange: 'reversed' }, margin: { l: 90, r: 8, t: 4, b: 30 } });
    $('#cp-vx-levels', root).textContent = `Typical VIX in each state: ${d.names.map((n, i) => `${n} ${d.levels[i].toFixed(1)}`).join(', ')}.`;
  }
})();

// ================================================================ view: backtest
const VIEW_backtest = (() => {
  const state = { train_end: 1989, rule: 'bull', threshold: 0.5, cost_bps: 0, cash_rate: 0 };

  return {
    mount(root) {
      root.innerHTML = `
        <div class="view-head">
          <div>
            <h2>Does knowing the regime pay?</h2>
            <p>The HMM is trained only on the years before the test starts. Each month it reads the real-time regime probabilities and decides next month's position. Change any setting and the server re-trains and re-runs the test.</p>
          </div>
          <span class="rubric">Rubric: comparative analysis and interpretation</span>
        </div>

        <div class="controls">
          <div class="field"><span>Rule</span><div class="seg" id="bt-rule"><button data-v="bull">Ride the Bull</button><button data-v="avoid_crisis">Avoid Crisis</button></div></div>
          <label class="field"><span>Train up to <output id="bt-te-o"></output></span><input type="range" id="bt-te" min="1930" max="2010" step="1" value="1989"></label>
          <label class="field"><span>Probability threshold <output id="bt-th-o"></output></span><input type="range" id="bt-th" min="0.2" max="0.8" step="0.05" value="0.5"></label>
          <label class="field"><span>Trading cost <output id="bt-c-o"></output></span><input type="range" id="bt-c" min="0" max="50" step="5" value="0"></label>
          <label class="field"><span>Cash interest <output id="bt-r-o"></output></span><input type="range" id="bt-r" min="0" max="0.06" step="0.005" value="0"></label>
        </div>

        <div class="kpis panel" id="bt-kpis"></div>

        <div class="panel" id="bt-main">
          <div class="panel-head"><h3 id="bt-title">Growth of $1</h3><span class="compute" id="bt-chip"></span></div>
          <p class="sub" id="bt-rule-text"></p>
          <div class="plot" id="bt-eq" style="min-height:560px"></div>
        </div>

        <div class="grid g-main">
          <div class="panel">
            <h3>Results</h3>
            <div class="scroll-x"><table id="bt-table"></table></div>
            <p class="note" style="margin:0">CAGR is the yearly growth rate. Sharpe is return per unit of risk. Max drawdown is the worst fall from a peak.</p>
          </div>
          <div class="panel">
            <h3>What the model learned before the test</h3>
            <div class="scroll-x"><table id="bt-train"></table></div>
            <p class="note" style="margin:0">No month after the training window is used to fit the model, and each month's decision uses only data up to that month.</p>
          </div>
        </div>`;

      segmented($('#bt-rule', root), state.rule, (v) => { state.rule = v; run(root); });
      const bind = (id, key, fmt) => {
        const el = $(id, root);
        rangeField(el, $(id + '-o', root), fmt);
        el.addEventListener('change', () => { state[key] = +el.value; run(root); });
      };
      bind('#bt-te', 'train_end', (v) => v);
      bind('#bt-th', 'threshold', (v) => Number(v).toFixed(2));
      bind('#bt-c', 'cost_bps', (v) => `${v} bp per switch`);
      bind('#bt-r', 'cash_rate', (v) => `${(v * 100).toFixed(1)}% / yr`);
      run(root);
    },
  };

  async function run(root) {
    const call = await withBusy($('#bt-main', root), `Training on 1871–${state.train_end} and testing…`, () => api('backtest', state));
    const d = call.data;
    computeChip($('#bt-chip', root), call);
    const b = d.stats.buy_hold, s = d.stats.strategy;
    $('#bt-title', root).textContent = `Growth of $1, ${d.test_window[0]}–${d.test_window[1]} (out of sample)`;
    $('#bt-rule-text', root).textContent = `${d.rule_text}. Trained on ${d.train_window[0]}–${d.train_window[1]} only.`;
    $('#bt-kpis', root).innerHTML = [
      [`${pct(b.max_drawdown)} → ${pct(s.max_drawdown)}`, 'worst drawdown: buy & hold → strategy'],
      [`${num(b.sharpe)} → ${num(s.sharpe)}`, 'Sharpe ratio'],
      [`${pct(b.cagr, 1)} → ${pct(s.cagr, 1)}`, 'yearly growth (CAGR)'],
      [pct(s.time_invested), `of months invested · ${s.trades} switches`],
    ].map(([v, l]) => `<div class="kpi"><span class="v">${v}</span><span class="l">${esc(l)}</span></div>`).join('');

    const x = d.series.dates;
    const probKey = d.rule === 'bull' ? 'p_bull' : 'p_crisis';
    const probName = d.rule === 'bull' ? 'P(Bull)' : 'P(Crisis)';
    const probCol = d.rule === 'bull' ? '#20c997' : '#ff6b3d';
    plot($('#bt-eq', root), [
      { x, y: d.series.buy_hold, name: 'Buy & hold', line: { color: '#8a94a6', width: 1.8 }, hovertemplate: '$%{y:.2f}<extra>Buy & hold</extra>' },
      { x, y: d.series.strategy, name: 'HMM strategy', line: { color: '#2b63d9', width: 2.6 }, hovertemplate: '$%{y:.2f}<extra>HMM strategy</extra>' },
      { x, y: d.series.dd_buy_hold, name: 'Buy & hold drawdown', yaxis: 'y2', fill: 'tozeroy', line: { color: '#8a94a6', width: 1 }, fillcolor: 'rgba(138,148,166,0.25)', hovertemplate: '%{y:.0%}<extra>B&H drawdown</extra>', showlegend: false },
      { x, y: d.series.dd_strategy, name: 'Strategy drawdown', yaxis: 'y2', fill: 'tozeroy', line: { color: '#2b63d9', width: 1 }, fillcolor: 'rgba(43,99,217,0.25)', hovertemplate: '%{y:.0%}<extra>strategy drawdown</extra>', showlegend: false },
      { x, y: d.series[probKey], name: probName, yaxis: 'y3', fill: 'tozeroy', line: { color: probCol, width: 1 }, fillcolor: probCol + '99', hovertemplate: `${probName} %{y:.0%}<extra></extra>`, showlegend: false },
    ], {
      hovermode: 'x unified', xaxis: { type: 'date', anchor: 'y3' },
      yaxis: { type: 'log', domain: [0.5, 1], title: { text: 'growth of $1 (log)' }, tickvals: [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500], ticktext: ['$0.5', '$1', '$2', '$5', '$10', '$20', '$50', '$100', '$200', '$500'] },
      yaxis2: { domain: [0.24, 0.44], tickformat: '.0%', title: { text: 'drawdown' } },
      yaxis3: { domain: [0, 0.18], range: [0, 1], tickformat: '.0%', title: { text: probName } },
      shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y3', y0: d.threshold, y1: d.threshold, line: { color: '#132033', dash: 'dot', width: 1 } }],
      margin: { l: 70, r: 12, t: 30, b: 30 },
    });

    const rows = [['Buy & hold', b], ['HMM strategy', s]];
    const cols = [['cagr', 'CAGR', (v) => pct(v, 1), 'max'], ['vol', 'Volatility', (v) => pct(v, 1), 'min'], ['sharpe', 'Sharpe', (v) => num(v), 'max'], ['max_drawdown', 'Max drawdown', (v) => pct(v), 'max'], ['worst_month', 'Worst month', (v) => pct(v, 1), 'max'], ['final', '$1 grows to', (v) => '$' + num(v, 2), 'max']];
    $('#bt-table', root).innerHTML = `<tr><th>Strategy</th>${cols.map((c) => `<th>${c[1]}</th>`).join('')}</tr>` +
      rows.map(([n, st]) => `<tr><td><b>${n}</b></td>${cols.map(([k, , f, dir]) => { const other = rows.find((r) => r[1] !== st)[1][k]; const win = dir === 'max' ? st[k] > other : st[k] < other; return `<td class="${win ? 'best' : ''}">${f(st[k])}</td>`; }).join('')}</tr>`).join('');
    $('#bt-train', root).innerHTML = '<tr><th>State</th><th>Return / yr</th><th>Volatility / yr</th></tr>' +
      d.train_states.map((t) => `<tr><td>${esc(t.name)}</td><td>${pct(t.ann_return, 1)}</td><td>${pct(t.ann_vol, 1)}</td></tr>`).join('');
  }
})();

// ================================================================ view: fusion
const VIEW_fusion = (() => {
  const state = { algo: 'kmeans', k: 4 };

  return {
    mount(root) {
      root.innerHTML = `
        <div class="view-head">
          <div>
            <h2>Fusion: what to hold, and when</h2>
            <p>The two halves of the project meet here. A daily 2-state HMM labels each trading day of 2013–2018 Calm or Stressed. Then we ask how each stock cluster performed in each kind of day.</p>
          </div>
          <span class="rubric">Rubric: result interpretation and innovation</span>
        </div>

        <div class="controls">
          <div class="field"><span>Clusters from</span><div class="seg" id="fu-algo"><button data-v="kmeans">K-Means</button><button data-v="gmm">Gaussian mixture</button><button data-v="agglomerative">Agglomerative</button></div></div>
          <label class="field"><span>k = <output id="fu-k-o"></output></span><input type="range" id="fu-k" min="2" max="8" value="4"></label>
          <span class="spacer"></span>
          <span class="compute" id="fu-chip"></span>
        </div>

        <div class="panel" id="fu-main">
          <h3 id="fu-insight">…</h3>
          <div class="grid g-main">
            <div class="plot tall" id="fu-bars"></div>
            <div class="scroll-x"><table id="fu-table"></table></div>
          </div>
        </div>

        <div class="panel">
          <h3>When the market was stressed</h3>
          <p class="sub" id="fu-sub">Equal-weight index of the 477 stocks. Shaded days are the ones the daily HMM marked as Stressed.</p>
          <div class="plot" id="fu-mkt"></div>
        </div>`;
      segmented($('#fu-algo', root), state.algo, (v) => { state.algo = v; run(root); });
      const k = $('#fu-k', root);
      rangeField(k, $('#fu-k-o', root));
      k.addEventListener('change', () => { state.k = +k.value; run(root); });
      run(root);
    },
  };

  async function run(root) {
    const call = await withBusy($('#fu-main', root), 'Combining clusters with regimes…', () => api('fusion', state));
    const d = call.data;
    computeChip($('#fu-chip', root), call);
    $('#fu-insight', root).textContent = d.insight;
    const names = d.rows.map((r) => r.cluster);
    plot($('#fu-bars', root), [
      { type: 'bar', name: 'Calm days', x: names, y: d.rows.map((r) => r.calm.ann_return), marker: { color: '#20c997' }, text: d.rows.map((r) => spct(r.calm.ann_return, 0)), textposition: 'outside', hovertemplate: '%{x}<br>calm: %{y:.1%} a year<extra></extra>' },
      { type: 'bar', name: 'Stressed days', x: names, y: d.rows.map((r) => r.stressed.ann_return), marker: { color: '#ff6b3d' }, text: d.rows.map((r) => spct(r.stressed.ann_return, 0)), textposition: 'outside', hovertemplate: '%{x}<br>stressed: %{y:.1%} a year<extra></extra>' },
    ], { barmode: 'group', yaxis: { tickformat: '.0%', title: { text: 'annualised return' } }, margin: { l: 64, r: 8, t: 30, b: 40 } });
    $('#fu-table', root).innerHTML = '<tr><th>Cluster</th><th>Stocks</th><th>Calm</th><th>Stressed</th></tr>' +
      d.rows.map((r) => `<tr><td><i class="dot" style="background:${r.color}"></i>${esc(r.cluster)}</td><td>${r.n}</td><td>${spct(r.calm.ann_return)}</td><td>${spct(r.stressed.ann_return)}</td></tr>`).join('');
    const m = d.market;
    $('#fu-sub', root).textContent = `Equal-weight index of the 477 stocks. Shaded days (${pct(d.stressed_share)} of all days) are the ones the daily HMM marked as Stressed.`;
    plot($('#fu-mkt', root), [{ x: m.dates, y: m.index, mode: 'lines', line: { color: '#132033', width: 1.6 }, hovertemplate: '%{x|%d %b %Y}: %{y:.2f}<extra></extra>' }],
      { shapes: stateBands(m.dates, m.stressed, ['#ffffff', '#ff6b3d'], 0.28).filter((s) => s.fillcolor.indexOf('255,255,255') < 0), yaxis: { title: { text: 'growth of $1' } }, xaxis: { type: 'date' }, margin: { l: 56, r: 8, t: 10, b: 30 } });
  }
})();

// ================================================================ router

const VIEWS = { overview: VIEW_overview, clustering: VIEW_clustering, hmm: VIEW_hmm, lab: VIEW_lab, compare: VIEW_compare, backtest: VIEW_backtest, fusion: VIEW_fusion };
const mounted = {};
const main = document.getElementById('main');

function route() {
  const id = (location.hash || '#overview').slice(1);
  const key = VIEWS[id] ? id : 'overview';
  $$('.nav a').forEach((a) => a.setAttribute('aria-current', a.getAttribute('href') === '#' + key ? 'page' : 'false'));
  for (const [k, sec] of Object.entries(mounted)) sec.hidden = k !== key;
  if (!mounted[key]) {
    const sec = document.createElement('section');
    sec.className = 'view';
    sec.id = 'view-' + key;
    main.appendChild(sec);
    mounted[key] = sec;
    VIEWS[key].mount(sec);
  } else {
    // charts drawn while hidden need a resize once visible
    requestAnimationFrame(() => {
      mounted[key].querySelectorAll('.js-plotly-plot').forEach((p) => window.Plotly && Plotly.Plots.resize(p));
      VIEWS[key].onShow && VIEWS[key].onShow();
    });
  }
  main.focus({ preventScroll: true });
  window.scrollTo(0, 0);
}

window.addEventListener('hashchange', route);
route();

async function pollHealth() {
  const el = document.getElementById('status');
  try {
    const { data } = await api('health', {}, { useCache: false });
    el.classList.toggle('ok', true);
    el.querySelector('span').textContent = 'Python model online';
  } catch (e) {
    el.classList.remove('ok');
    el.querySelector('span').textContent = 'Model server not reachable';
    setTimeout(pollHealth, 5000);
  }
}
pollHealth();
