/* ==========================================================
   VinBot — Frontend App Logic
   ========================================================== */

const API_BASE = window.location.origin;

// ── State ─────────────────────────────────────────────────
let sessionId = null;
let pendingFilters = {}; // chips selected before sending
let shownWineIds = [];   // track shown wines to avoid duplicates

// ── DOM refs ──────────────────────────────────────────────
const chatWindow    = document.getElementById('chatWindow');
const userInput     = document.getElementById('userInput');
const sendBtn       = document.getElementById('sendBtn');
const cardsPanel    = document.getElementById('cardsPanel');
const cardsGrid     = document.getElementById('cardsGrid');
const cardsSubtitle = document.getElementById('cardsSubtitle');
const resetBtn      = document.getElementById('resetBtn');
const loadingOverlay= document.getElementById('loadingOverlay');
const menuToggle    = document.getElementById('menuToggle');
const sidebar       = document.getElementById('sidebar');
const cardTemplate  = document.getElementById('wineCardTemplate');

// ── Sidebar toggle (mobile) ────────────────────────────────
menuToggle.addEventListener('click', () => sidebar.classList.toggle('open'));
document.addEventListener('click', e => {
  if (window.innerWidth <= 860 && !sidebar.contains(e.target) && e.target !== menuToggle) {
    sidebar.classList.remove('open');
  }
});

// ── Chip logic ─────────────────────────────────────────────
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const type = chip.dataset.type;
    const food = chip.dataset.food;
    const body = chip.dataset.body;

    if (type) {
      // Exclusive within type chips
      document.querySelectorAll('[data-type]').forEach(c => c.classList.remove('active'));
      chip.classList.toggle('active');
      if (chip.classList.contains('active')) {
        pendingFilters.wine_type = type;
        prefillInput(`I'd like a ${type} wine`);
      } else {
        delete pendingFilters.wine_type;
      }
    }

    if (food) {
      chip.classList.toggle('active');
      if (!pendingFilters.food) pendingFilters.food = [];
      if (chip.classList.contains('active')) {
        pendingFilters.food.push(food);
        prefillInput(`I want something that pairs well with ${pendingFilters.food.join(', ')}`);
      } else {
        pendingFilters.food = pendingFilters.food.filter(f => f !== food);
      }
    }

    if (body) {
      document.querySelectorAll('[data-body]').forEach(c => c.classList.remove('active'));
      chip.classList.toggle('active');
      if (chip.classList.contains('active')) {
        pendingFilters.body = body;
        prefillInput(`I prefer a ${body} wine`);
      } else {
        delete pendingFilters.body;
      }
    }
  });
});

function prefillInput(text) {
  userInput.value = text;
  userInput.dispatchEvent(new Event('input'));
  userInput.focus();
}

// ── Auto-resize textarea ───────────────────────────────────
userInput.addEventListener('input', () => {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 160) + 'px';
});

// ── Keyboard send ─────────────────────────────────────────
userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});
sendBtn.addEventListener('click', handleSend);

// ── Reset ─────────────────────────────────────────────────
resetBtn.addEventListener('click', async () => {
  if (sessionId) {
    await fetch(`${API_BASE}/api/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    }).catch(() => {});
  }
  sessionId = null;
  shownWineIds = [];
  pendingFilters = {};
  document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
  cardsPanel.style.display = 'none';
  cardsGrid.innerHTML = '';
  chatWindow.innerHTML = '';
  addBotMessage(
    'Fresh start! 🍷 Tell me what kind of wine you\'re looking for — ' +
    'occasion, food, region, or just a mood. I\'m here to help!'
  );
  userInput.value = '';
  userInput.style.height = 'auto';
});

// ── Main send handler ─────────────────────────────────────
async function handleSend() {
  const text = userInput.value.trim();
  if (!text) return;

  addUserMessage(text);
  userInput.value = '';
  userInput.style.height = 'auto';

  // Show typing indicator
  const typingId = addTypingIndicator();

  try {
    const payload = { message: text };
    if (sessionId) payload.session_id = sessionId;

    const res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    sessionId = data.session_id;

    removeTypingIndicator(typingId);
    addBotMessage(data.reply);

    if (data.wines && data.wines.length > 0) {
      renderWineCards(data.wines);
      // Track shown IDs
      data.wines.forEach(w => {
        if (w.WineID && !shownWineIds.includes(w.WineID)) {
          shownWineIds.push(w.WineID);
        }
      });
    }

    // Clear pending filters after send
    pendingFilters = {};
    clearChips();

  } catch (err) {
    removeTypingIndicator(typingId);
    const msg = err.message || '';
    if (msg.includes('429') || msg.toLowerCase().includes('quota') || msg.toLowerCase().includes('rate')) {
      addBotMessage(
        '⏳ I\'m getting too many requests right now (free-tier rate limit). ' +
        'Please wait a few seconds and try again — your message wasn\'t lost!'
      );
    } else {
      addBotMessage(`⚠️ Something went wrong. Please try again in a moment.`);
    }
    console.error('[VinBot error]', err);
  }
}

// ── Message rendering ─────────────────────────────────────
function addUserMessage(text) {
  const msg = document.createElement('div');
  msg.className = 'message user-message';
  msg.innerHTML = `
    <div class="avatar user-avatar">👤</div>
    <div class="bubble user-bubble">${escapeHtml(text)}</div>
  `;
  chatWindow.appendChild(msg);
  scrollToBottom();
}

function addBotMessage(text) {
  const msg = document.createElement('div');
  msg.className = 'message bot-message';
  const formatted = formatBotText(text);
  msg.innerHTML = `
    <div class="avatar bot-avatar">🍷</div>
    <div class="bubble bot-bubble">${formatted}</div>
  `;
  chatWindow.appendChild(msg);
  scrollToBottom();
  return msg;
}

function addTypingIndicator() {
  const id = 'typing-' + Date.now();
  const msg = document.createElement('div');
  msg.className = 'message bot-message';
  msg.id = id;
  msg.innerHTML = `
    <div class="avatar bot-avatar">🍷</div>
    <div class="bubble bot-bubble">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;
  chatWindow.appendChild(msg);
  scrollToBottom();
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// ── Text formatting ────────────────────────────────────────
function formatBotText(text) {
  // Convert markdown-ish formatting
  let html = escapeHtml(text);

  // Bold **text**
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic *text*
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Headers ### text
  html = html.replace(/^###\s+(.+)$/gm, '<h4 style="margin:0.6rem 0 0.3rem;color:var(--gold);font-family:Playfair Display,serif;font-size:0.9rem;">$1</h4>');
  html = html.replace(/^##\s+(.+)$/gm, '<h3 style="margin:0.7rem 0 0.3rem;color:var(--gold);font-family:Playfair Display,serif;font-size:1rem;">$1</h3>');
  // Bullet points
  html = html.replace(/^[•·]\s+(.+)$/gm, '<li style="margin-left:1rem;list-style:disc;">$1</li>');
  html = html.replace(/(<li[^>]*>.*<\/li>)/gs, '<ul style="padding-left:0.5rem;margin:0.3rem 0;">$1</ul>');
  // Newlines → <br> (but not inside tags)
  html = html.replace(/\n/g, '<br>');

  return html;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Wine card rendering ────────────────────────────────────
function renderWineCards(wines) {
  cardsGrid.innerHTML = '';

  wines.forEach((wine, idx) => {
    const card = cardTemplate.content.cloneNode(true).querySelector('.wine-card');
    card.style.animationDelay = `${idx * 0.07}s`;

    // Type badge
    card.querySelector('.card-type-badge').textContent = wine.Type || '—';

    // Name
    card.querySelector('.card-name').textContent = wine.WineName || 'Unknown Wine';

    // Winery
    card.querySelector('.card-winery').textContent = wine.WineryName || '';

    // Meta tags
    card.querySelector('.country-tag').textContent = `🌍 ${wine.Country || '—'}`;
    card.querySelector('.body-tag').textContent     = wine.Body || '—';
    card.querySelector('.abv-tag').textContent      = wine.ABV ? `${wine.ABV}% ABV` : '—';

    // Score bar
    const score = wine.similarity_score || 0;
    const pct = Math.round(score * 100);
    card.querySelector('.score-bar').style.width = `${Math.min(pct * 1.2, 100)}%`;
    card.querySelector('.score-label').textContent = `${pct}% match`;

    // Expanded details
    const grapes = wine.grapes_parsed?.join(', ') || cleanList(wine.Grapes);
    const food   = wine.food_parsed?.join(', ')   || cleanList(wine.Harmonize);
    card.querySelector('.grapes-val').textContent  = grapes || '—';
    card.querySelector('.food-val').textContent    = food   || '—';
    card.querySelector('.acidity-val').textContent = wine.Acidity || '—';
    card.querySelector('.region-val').textContent  = wine.RegionName || '—';

    // Expand toggle
    const detailsEl = card.querySelector('.card-details');
    const expandBtn = card.querySelector('.card-expand-btn');
    expandBtn.addEventListener('click', () => {
      const expanded = detailsEl.classList.contains('expanded');
      detailsEl.classList.toggle('collapsed', expanded);
      detailsEl.classList.toggle('expanded', !expanded);
      expandBtn.textContent = expanded ? 'Show details ▾' : 'Hide details ▴';
    });

    cardsGrid.appendChild(card);
  });

  cardsPanel.style.display = 'block';
  cardsSubtitle.textContent = `${wines.length} wine${wines.length !== 1 ? 's' : ''} matched your preferences`;
  cardsPanel.scrollTop = 0;
}

function cleanList(str) {
  if (!str) return '';
  return str.replace(/[\[\]'"]/g, '').replace(/,\s*/g, ', ');
}

function clearChips() {
  document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
}

// ── Init ──────────────────────────────────────────────────
// Nothing to do — welcome message is in the HTML
