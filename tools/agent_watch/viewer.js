/* ARC Agent Watch viewer. Classic script, no build step, no network.
 * Pure logic lives at the top and is exported for `node --test`; the DOM code
 * at the bottom only runs when a `document` exists. All trace text is untrusted:
 * it is only ever put into the page with textContent. */
(function () {
  'use strict';

  var PALETTE = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'];
  var TASK_EVENTS = ['task_start', 'turn', 'tool_result', 'submit', 'rescue', 'attempt_end', 'task_end'];
  var KNOWN_EVENTS = TASK_EVENTS.concat(['run_start', 'heartbeat', 'run_end']);
  var COLORS = {
    think: '#8E7CF8', python: '#6CB6FF', submit: '#FFC857', rescue: '#FF851B',
    ok: '#4ADE80', bad: '#FF6B6B', grey: '#8793A1', dim: '#2A313B'
  };
  var MAX_GAP_S = 1.5;
  var MIN_DELAY_MS = 30;

  function num(v) { return typeof v === 'number' && isFinite(v) ? v : 0; }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  /* JSONL, line by line. A last line that does not parse is a truncated write and
   * is dropped silently; other bad lines are counted. Unknown `ev` kept out. */
  function parseJsonl(text) {
    var lines = String(text).split('\n');
    var last = -1;
    var i;
    for (i = lines.length - 1; i >= 0; i--) {
      if (lines[i].trim()) { last = i; break; }
    }
    var events = [];
    var bad = 0, unknown = 0, truncated = false;
    for (i = 0; i <= last; i++) {
      var line = lines[i].trim();
      if (!line) continue;
      var obj;
      try { obj = JSON.parse(line); } catch (err) {
        if (i === last) truncated = true; else bad++;
        continue;
      }
      if (!obj || typeof obj !== 'object' || Array.isArray(obj) || typeof obj.ev !== 'string') { bad++; continue; }
      if (KNOWN_EVENTS.indexOf(obj.ev) < 0) { unknown++; continue; }
      events.push(obj);
    }
    return { events: events, bad: bad, unknown: unknown, truncated: truncated };
  }

  function cfgOf(e) { return typeof e.cfg === 'string' && e.cfg ? e.cfg : 'default'; }

  function listConfigs(events) {
    var seen = [];
    events.forEach(function (e) { var c = cfgOf(e); if (seen.indexOf(c) < 0) seen.push(c); });
    return seen;
  }

  /* Group one config's events: run_start info, tasks (listed ids first, so queued
   * ones appear), heartbeats, last t. */
  function buildRun(events, cfg) {
    var evs = cfg == null ? events : events.filter(function (e) { return cfgOf(e) === cfg; });
    var run = { cfg: cfg, start: null, end: null, heartbeats: [], tasks: [], lastT: 0 };
    var map = {};
    function get(id) {
      if (!map[id]) { map[id] = { id: id, start: null, events: [] }; run.tasks.push(map[id]); }
      return map[id];
    }
    evs.forEach(function (e) {
      if (e.ev === 'run_start' && !run.start) {
        run.start = e;
        (Array.isArray(e.tasks) ? e.tasks : []).forEach(function (id) { if (typeof id === 'string') get(id); });
      }
    });
    evs.forEach(function (e) {
      if (typeof e.t === 'number' && isFinite(e.t) && e.t > run.lastT) run.lastT = e.t;
      if (e.ev === 'run_end') run.end = e;
      else if (e.ev === 'heartbeat') run.heartbeats.push(e);
      else if (TASK_EVENTS.indexOf(e.ev) >= 0 && typeof e.task === 'string') {
        var t = get(e.task);
        t.events.push(e);
        if (e.ev === 'task_start' && !t.start) t.start = e;
      }
    });
    return run;
  }

  /* verified | unverified | running | queued, from attempt_end / task_end. */
  function taskStatus(evs) {
    if (!evs.length) return 'queued';
    var ends = [], started = {}, taskEnd = null, nStarted = 0;
    evs.forEach(function (e) {
      if (e.ev === 'attempt_end') ends.push(e);
      if (e.ev === 'task_end') taskEnd = e;
      if (e.ev === 'turn' || e.ev === 'submit' || e.ev === 'tool_result' || e.ev === 'rescue' || e.ev === 'attempt_end') {
        var a = Number.isInteger(e.att) ? e.att : 0;
        if (!started[a]) { started[a] = true; nStarted++; }
      }
    });
    var anyVerified = ends.some(function (e) { return e.verified === true; });
    if (taskEnd) return anyVerified || num(taskEnd.verified) > 0 ? 'verified' : 'unverified';
    if (anyVerified) return 'verified';
    if (nStarted > 0 && ends.length >= nStarted) return 'unverified';
    return 'running';
  }

  function cardKey(att, turn, name) { return att + '|' + turn + '|' + name; }

  /* The whole replay state of one task as of event index `idx` (inclusive;
   * null/undefined = everything). */
  function stateAt(task, idx) {
    var all = task.events;
    var n = idx == null ? all.length : Math.max(0, Math.min(all.length, idx + 1));
    var evs = all.slice(0, n);
    var cards = [], pendRes = {}, pendSub = {}, progNo = {};
    var tokens = 0, latestSubmit = null, lastTurn = null, taskEnd = null, lastT = 0, lastAtt = 0, maxTok = 0;
    evs.forEach(function (e) {
      if (typeof e.t === 'number') lastT = e.t;
      var att = Number.isInteger(e.att) ? e.att : 0;
      var turn = e.turn;
      var q, card;
      if (e.ev === 'turn') {
        lastAtt = att;
        tokens += num(e.completion_tokens);
        maxTok = Math.max(maxTok, num(e.completion_tokens));
        lastTurn = { att: att, turn: turn, finish: e.finish };
        cards.push({ kind: 'think', att: att, turn: turn, ev: e });
        (Array.isArray(e.tool_calls) ? e.tool_calls : []).forEach(function (call) {
          var name = call && call.name;
          card = { kind: name === 'python' ? 'python' : name === 'submit' ? 'submit' : 'tool', att: att, turn: turn, name: name, call: call || {}, result: null, submit: null };
          cards.push(card);
          (pendRes[cardKey(att, turn, name)] = pendRes[cardKey(att, turn, name)] || []).push(card);
          if (card.kind === 'submit') (pendSub[att + '|' + turn] = pendSub[att + '|' + turn] || []).push(card);
        });
      } else if (e.ev === 'tool_result') {
        q = pendRes[cardKey(att, turn, e.name)];
        card = q && q.shift();
        if (card) card.result = e;
      } else if (e.ev === 'submit') {
        q = pendSub[att + '|' + turn];
        card = q && q.shift();
        if (!card) { card = { kind: 'submit', att: att, turn: turn, name: 'submit', call: {}, result: null, submit: null }; cards.push(card); }
        card.submit = e;
        progNo[att] = (progNo[att] || 0) + 1;
        card.progNo = progNo[att];
        latestSubmit = e;
      } else if (e.ev === 'rescue') {
        cards.push({ kind: 'rescue', att: att, turn: turn, ev: e });
      } else if (e.ev === 'attempt_end') {
        cards.push({ kind: 'end', att: att, ev: e });
      } else if (e.ev === 'task_end') {
        taskEnd = e;
      }
    });
    var status = taskStatus(evs);
    var thinkingNow = null;
    if (status === 'running') {
      var last = cards[cards.length - 1];
      var pending = last && (last.kind === 'python' || last.kind === 'submit' || last.kind === 'tool') && !last.result;
      if (!pending) {
        thinkingNow = { att: lastAtt, turn: lastTurn && lastTurn.att === lastAtt ? (lastTurn.turn || 0) + 1 : 1, since: lastT };
      }
    }
    return { events: evs, count: n, cards: cards, status: status, latestSubmit: latestSubmit, tokens: tokens,
      maxTokens: maxTok, thinkingNow: thinkingNow, taskEnd: taskEnd, t: lastT, lastTurn: lastTurn };
  }

  function gridDims(g) { return Array.isArray(g) && g.length && Array.isArray(g[0]) ? [g.length, g[0].length] : [0, 0]; }

  /* Wrong cells between expected and got: [[r,c,got,expected], ...]. Cells outside
   * the other grid's shape count as wrong with null on that side. */
  function gridDiff(expected, got) {
    var out = { cells: [], rows: 0, cols: 0, shapeMismatch: false, mask: {} };
    if (!Array.isArray(expected) || !Array.isArray(got)) return out;
    var de = gridDims(expected), dg = gridDims(got);
    out.rows = Math.max(de[0], dg[0]); out.cols = Math.max(de[1], dg[1]);
    out.shapeMismatch = de[0] !== dg[0] || de[1] !== dg[1];
    for (var r = 0; r < out.rows; r++) {
      for (var c = 0; c < out.cols; c++) {
        var ev = r < de[0] && c < de[1] ? expected[r][c] : null;
        var gv = r < dg[0] && c < dg[1] ? got[r][c] : null;
        if (ev !== gv) { out.cells.push([r, c, gv, ev]); out.mask[r + ',' + c] = true; }
      }
    }
    return out;
  }

  function countPairs(cells) {
    var m = {};
    cells.forEach(function (c) {
      var k = c[2] + '>' + c[3];
      m[k] = m[k] || { got: c[2], expected: c[3], n: 0 };
      m[k].n++;
    });
    return Object.keys(m).map(function (k) { return m[k]; }).sort(function (a, b) { return b.n - a.n; });
  }

  /* Which train example to show for a submit: the first failing one, else 0. */
  function pickSubmitExample(sub) {
    if (sub && Array.isArray(sub.wrong) && sub.wrong.length && Number.isInteger(sub.wrong[0].ex)) return sub.wrong[0].ex;
    return 0;
  }

  function submitDetail(sub) {
    if (!sub) return '';
    if (sub.verified === true) return 'All demonstrations reproduced.';
    var w = Array.isArray(sub.wrong) ? sub.wrong : [];
    if (!w.length) return Array.isArray(sub.outputs) && sub.outputs.some(function (o) { return o == null; }) ? 'The program produced no output for some examples.' : '';
    return w.slice(0, 3).map(function (x) {
      return 'Example ' + (num(x.ex) + 1) + ': ' + num(x.total) + ' cells wrong.';
    }).join(' ');
  }

  function excerpt(text, n) {
    var s = String(text == null ? '' : text).replace(/\s+/g, ' ').trim();
    n = n || 300;
    return s.length > n ? s.slice(0, n).trimEnd() + '…' : s;
  }

  function formatClock(sec) {
    var s = Math.max(0, Math.floor(num(sec)));
    function p(x) { return (x < 10 ? '0' : '') + x; }
    return p(Math.floor(s / 3600)) + ':' + p(Math.floor((s % 3600) / 60)) + ':' + p(s % 60);
  }

  function formatInt(n) { return Math.round(num(n)).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ','); }

  function formatDur(sec) {
    var s = num(sec);
    if (s < 60) return s.toFixed(s < 10 ? 1 : 0) + ' s';
    return Math.floor(s / 60) + 'm ' + Math.round(s % 60) + 's';
  }

  function computeTotals(run) {
    var tokens = 0, verified = 0, tokS = null;
    run.tasks.forEach(function (t) {
      t.events.forEach(function (e) { if (e.ev === 'turn') tokens += num(e.completion_tokens); });
      if (taskStatus(t.events) === 'verified') verified++;
    });
    run.heartbeats.forEach(function (h) { if (typeof h.tok_s === 'number') tokS = h.tok_s; });
    return { elapsed: run.lastT, tokens: tokens, tokS: tokS, verified: verified, tasks: run.tasks.length };
  }

  function heartbeatSeries(run) {
    return run.heartbeats.filter(function (h) { return typeof h.tok_s === 'number' && typeof h.t === 'number'; })
      .map(function (h) { return { t: h.t, v: h.tok_s }; });
  }

  function tickColor(e) {
    switch (e.ev) {
      case 'turn': return COLORS.think;
      case 'tool_result': return e.name === 'submit' ? '#B08A2E' : COLORS.python;
      case 'submit': return COLORS.submit;
      case 'rescue': return COLORS.rescue;
      case 'attempt_end': case 'task_end': return e.verified === true || num(e.verified) > 0 ? COLORS.ok : COLORS.bad;
      default: return COLORS.grey;
    }
  }

  /* Milliseconds to wait between event i and i+1: the t gap over speed, capped. */
  function playbackDelayMs(tFrom, tTo, speed) {
    var gap = Math.max(0, num(tTo) - num(tFrom));
    var s = Math.min(gap / (speed > 0 ? speed : 1), MAX_GAP_S);
    return Math.max(MIN_DELAY_MS, Math.round(s * 1000));
  }

  var api = {
    PALETTE: PALETTE, SPEEDS: [1, 2, 4, 8, 32], parseJsonl: parseJsonl, listConfigs: listConfigs, buildRun: buildRun,
    taskStatus: taskStatus, stateAt: stateAt, gridDiff: gridDiff, countPairs: countPairs,
    pickSubmitExample: pickSubmitExample, submitDetail: submitDetail, excerpt: excerpt, formatClock: formatClock,
    formatInt: formatInt, formatDur: formatDur, computeTotals: computeTotals, heartbeatSeries: heartbeatSeries,
    tickColor: tickColor, playbackDelayMs: playbackDelayMs, escapeHtml: escapeHtml
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof document === 'undefined') return;

  /* ------------------------------------------------------------------ DOM */

  var S = {
    events: [], cfg: null, run: null, taskId: null, idx: 0, speed: 8, playing: false, timer: null,
    tab: 0, expanded: {}, file: null, notes: ''
  };
  var $ = function (id) { return document.getElementById(id); };

  function h(tag, props) {
    var el = document.createElement(tag);
    var p = props || {};
    Object.keys(p).forEach(function (k) {
      var v = p[k];
      if (v == null || v === false) return;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else if (k === 'on') Object.keys(v).forEach(function (ev) { el.addEventListener(ev, v[ev]); });
      else el.setAttribute(k, v === true ? '' : String(v));
    });
    for (var i = 2; i < arguments.length; i++) {
      var kid = arguments[i];
      if (kid == null || kid === false) continue;
      el.appendChild(typeof kid === 'string' ? document.createTextNode(kid) : kid);
    }
    return el;
  }
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

  function gridEl(grid, opts) {
    opts = opts || {};
    var d = gridDims(grid);
    if (!d[0]) return h('div', { class: 'grid-missing', text: opts.missing || 'no grid' });
    var size = Math.max(4, Math.min(opts.max || 22, Math.floor((opts.box || 120) / Math.max(d[0], d[1]))));
    var el = h('div', { class: 'grid', role: 'img', 'aria-label': (opts.label || 'grid') + ' ' + d[0] + ' by ' + d[1] });
    el.style.gridTemplateColumns = 'repeat(' + d[1] + ', ' + size + 'px)';
    el.style.gridAutoRows = size + 'px';
    for (var r = 0; r < d[0]; r++) {
      for (var c = 0; c < d[1]; c++) {
        var cell = document.createElement('div');
        var v = grid[r][c];
        cell.style.background = opts.mask ? (opts.mask[r + ',' + c] ? '#FF6B6B' : '#232A33')
          : (PALETTE[v] || '#2A313B');
        el.appendChild(cell);
      }
    }
    return el;
  }
  function gridBlock(label, grid, opts) {
    var d = gridDims(grid);
    return h('div', { class: 'gblock' },
      h('div', { class: 'cap', text: label + (d[0] ? ' ' + d[0] + '×' + d[1] : '') }),
      gridEl(grid, opts));
  }

  function curTask() {
    if (!S.run) return null;
    for (var i = 0; i < S.run.tasks.length; i++) if (S.run.tasks[i].id === S.taskId) return S.run.tasks[i];
    return null;
  }

  var STATUS = {
    verified: [COLORS.ok, 'verified'], unverified: [COLORS.bad, 'unverified'],
    running: [COLORS.submit, 'running'], queued: [COLORS.grey, 'queued']
  };

  function renderHeader() {
    var run = S.run;
    var chip = clear($('runchip'));
    if (run) {
      var st = run.start || {};
      chip.appendChild(h('span', { class: 'muted', text: 'run ' }));
      chip.appendChild(document.createTextNode((st.run || '?') + ' / ' + (S.cfg || 'default')));
      var bits = [st.model, st.effort, st.mode].filter(Boolean).join(' · ');
      if (bits) chip.appendChild(h('span', { class: 'muted', text: '  ' + bits }));
      if (!run.end) chip.appendChild(h('span', { class: 'muted', text: '  (no run_end: live or crashed)' }));
    } else chip.textContent = 'no trace loaded';
    var cfgs = listConfigs(S.events);
    var sel = clear($('cfgsel'));
    sel.hidden = cfgs.length < 2;
    cfgs.forEach(function (c) {
      var o = h('option', { value: c, text: c });
      if (c === S.cfg) o.selected = true;
      sel.appendChild(o);
    });
    var tot = run ? computeTotals(run) : null;
    $('t-elapsed').textContent = tot ? formatClock(tot.elapsed) : '--';
    $('t-tokens').textContent = tot ? formatInt(tot.tokens) : '--';
    $('t-toks').textContent = tot && tot.tokS != null ? formatInt(tot.tokS) : '--';
    $('t-solved').textContent = tot ? tot.verified + ' / ' + tot.tasks : '--';
    $('status').textContent = S.notes;
  }

  function renderRight(state) {
    var run = S.run, box = clear($('right'));
    if (!run) return;
    var hb = run.heartbeats.length ? run.heartbeats[run.heartbeats.length - 1] : null;
    var series = heartbeatSeries(run);
    var health = h('section', { class: 'panel' }, h('h2', { text: 'Run health' }),
      h('div', { class: 'cap', text: 'AGGREGATE TOKENS/SEC' }));
    if (series.length > 1) {
      var W = 280, H = 64, tMin = series[0].t, tMax = series[series.length - 1].t || 1;
      var vMax = Math.max.apply(null, series.map(function (p) { return p.v; })) || 1;
      var pts = series.map(function (p) {
        return (((p.t - tMin) / ((tMax - tMin) || 1)) * W).toFixed(1) + ',' + (H - 4 - (p.v / vMax) * (H - 8)).toFixed(1);
      }).join(' ');
      var ns = 'http://www.w3.org/2000/svg';
      var svg = document.createElementNS(ns, 'svg');
      svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H); svg.setAttribute('width', '100%'); svg.setAttribute('height', String(H));
      svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Tokens per second over the run, peak ' + Math.round(vMax));
      var pl = document.createElementNS(ns, 'polyline');
      pl.setAttribute('points', pts); pl.setAttribute('fill', 'none'); pl.setAttribute('stroke', '#3DD9C0');
      pl.setAttribute('stroke-width', '2'); pl.setAttribute('stroke-linejoin', 'round');
      svg.appendChild(pl); health.appendChild(svg);
    } else health.appendChild(h('div', { class: 'muted small', text: series.length ? 'One heartbeat so far.' : 'No heartbeats in this trace.' }));
    function stat(label, v) { return h('div', null, h('div', { class: 'cap', text: label }), v == null ? '--' : String(v)); }
    health.appendChild(h('div', { class: 'stats mono' },
      stat('TOK/S', hb && hb.tok_s != null ? Math.round(hb.tok_s) : null),
      stat('RUNNING / WAITING', hb ? (hb.running == null ? '--' : hb.running) + ' / ' + (hb.waiting == null ? '--' : hb.waiting) : null),
      stat('KV CACHE', hb && hb.kv_pct != null ? Math.round(hb.kv_pct) + '%' : null),
      stat('GPU MEM', hb && hb.gpu_mem_mib != null ? formatInt(hb.gpu_mem_mib) + ' MiB' : null)));
    health.appendChild(h('div', { class: 'alert' },
      h('span', { class: 'dot', style: 'background:' + (run.end ? COLORS.grey : COLORS.ok) }),
      h('span', { text: run.end ? 'Run finished (run_end present)' : hb ? 'Last heartbeat at ' + formatClock(hb.t) : 'No heartbeat seen' })));
    box.appendChild(health);

    var list = h('div', { class: 'tasklist' });
    run.tasks.forEach(function (t) {
      var status = t.id === S.taskId && state ? state.status : taskStatus(t.events);
      var info = STATUS[status];
      list.appendChild(h('button', { class: 'taskrow', type: 'button', 'aria-current': t.id === S.taskId ? 'true' : null,
        on: { click: function () { selectTask(t.id); } } },
      h('span', { class: 'dot', style: 'background:' + info[0] }),
      h('span', { class: 'mono grow', text: t.id }),
      h('span', { class: 'muted small', text: info[1] })));
    });
    box.appendChild(h('section', { class: 'panel' },
      h('div', { class: 'row-between' }, h('h2', { text: 'Tasks in this config' }), h('span', { class: 'muted small', text: run.tasks.length + ' listed' })),
      list));
  }

  function renderLeft(task, state) {
    var box = clear($('left'));
    if (!task) return;
    var train = task.start && Array.isArray(task.start.train) ? task.start.train : [];
    var test = task.start && Array.isArray(task.start.test) ? task.start.test : [];
    var answers = task.start && Array.isArray(task.start.answers) ? task.start.answers : [];
    var tabs = [];
    train.forEach(function (_, i) { tabs.push({ label: 'Example ' + (i + 1), kind: 'train', i: i }); });
    test.forEach(function (_, i) { tabs.push({ label: test.length > 1 ? 'Test ' + (i + 1) : 'Test', kind: 'test', i: i }); });
    if (S.tab >= tabs.length) S.tab = 0;
    var panel = h('section', { class: 'panel' },
      h('div', { class: 'row-between' },
        h('h2', null, 'Task ', h('span', { class: 'mono', text: task.id })),
        h('span', { class: 'muted small', text: train.length + ' train · ' + test.length + ' test' })));
    if (!tabs.length) {
      panel.appendChild(h('p', { class: 'muted small', text: 'No task_start event yet: this task is queued.' }));
    } else {
      var tabRow = h('div', { class: 'tabs', role: 'group', 'aria-label': 'Puzzle examples' });
      tabs.forEach(function (tb, k) {
        tabRow.appendChild(h('button', { class: 'tab', type: 'button', 'aria-pressed': k === S.tab ? 'true' : 'false',
          on: { click: function () { S.tab = k; render(); } }, text: tb.label }));
      });
      panel.appendChild(tabRow);
      var cur = tabs[S.tab], inp, outp, outLabel = 'OUTPUT';
      if (cur.kind === 'train') { inp = train[cur.i] && train[cur.i].input; outp = train[cur.i] && train[cur.i].output; }
      else { inp = test[cur.i] && test[cur.i].input; outp = answers[cur.i]; outLabel = 'EXPECTED'; }
      panel.appendChild(h('div', { class: 'pair' },
        gridBlock('INPUT', inp, { box: 130, label: 'input' }), h('span', { class: 'arrow', 'aria-hidden': 'true', text: '→' }),
        gridBlock(outLabel, outp, { box: 130, label: 'output', missing: 'answer not known' })));
    }
    box.appendChild(panel);

    var sub = state && state.latestSubmit;
    var sp = h('section', { class: 'panel' });
    if (!sub) {
      sp.appendChild(h('h2', { text: 'Latest submit' }));
      sp.appendChild(h('p', { class: 'muted small', text: 'No submit yet at this point of the run.' }));
    } else {
      var ex = pickSubmitExample(sub);
      var exp = train[ex] ? train[ex].output : null;
      var outs = Array.isArray(sub.outputs) ? sub.outputs : [];
      var got = outs[ex] || null;
      var chipOk = sub.verified === true;
      sp.appendChild(h('div', { class: 'row-between' },
        h('h2', { text: 'Latest submit · turn ' + sub.turn }),
        h('span', { class: 'chip ' + (chipOk ? 'good' : 'badc'), text: num(sub.reproduced) + ' of ' + num(sub.n_train) + ' reproduced' })));
      sp.appendChild(h('div', { class: 'muted small', text: 'Showing Example ' + (ex + 1) }));
      var diff = gridDiff(exp, got);
      var cells = diff.cells;
      if (!cells.length && Array.isArray(sub.wrong)) {
        sub.wrong.forEach(function (w) { if (w.ex === ex && Array.isArray(w.cells)) cells = w.cells; });
      }
      sp.appendChild(h('div', { class: 'trio' },
        gridBlock('EXPECTED', exp, { box: 90, max: 16, label: 'expected', missing: 'unknown' }),
        gridBlock('PROGRAM OUTPUT', got, { box: 90, max: 16, label: 'program output', missing: 'no output' }),
        got && exp ? gridBlock('WRONG CELLS', exp, { box: 90, max: 16, label: 'wrong cells', mask: diff.mask }) : null));
      var lines = h('div', { class: 'wrong mono' });
      if (!got) lines.appendChild(h('div', { text: 'The program produced no grid for this example.' }));
      cells.slice(0, 10).forEach(function (c) {
        lines.appendChild(h('div', { text: '(' + c[0] + ',' + c[1] + ') got ' + c[2] + ' expected ' + c[3] }));
      });
      var pairs = countPairs(cells);
      if (pairs.length) lines.appendChild(h('div', { class: 'muted', text: 'pairs (got→expected): ' + pairs.slice(0, 6).map(function (p) { return p.got + '→' + p.expected + ' ×' + p.n; }).join(', ') }));
      if (cells.length > 10) lines.appendChild(h('div', { class: 'muted', text: '+' + (cells.length - 10) + ' more wrong cells' }));
      if (!cells.length && got) lines.appendChild(h('div', { class: 'muted', text: 'No wrong cells on this example.' }));
      sp.appendChild(lines);
      var flags = Array.isArray(sub.flags) ? sub.flags : [];
      var fl = h('div', { class: 'flags small' }, 'Plausibility flags on this program: ');
      if (!flags.length) fl.appendChild(h('span', { class: 'okc', text: 'none' }));
      else flags.forEach(function (f) { fl.appendChild(h('span', { class: 'flag', text: String(f) })); });
      sp.appendChild(fl);
    }
    box.appendChild(sp);
  }

  function marker(n, color, dashed) {
    var m = h('div', { class: 'marker' + (dashed ? ' dashed' : ''), text: dashed ? '' : String(n == null ? '' : n) });
    if (dashed) m.style.borderColor = color; else m.style.background = color;
    return m;
  }
  function row(n, color, kindLabel, meta, body, dashed) {
    var head = h('div', { class: 'cardhead' }, h('span', { class: 'kind', text: kindLabel }), meta ? h('span', { class: 'mono muted small', text: meta }) : null);
    head.firstChild.style.color = color;
    return h('div', { class: 'trow' }, h('div', { class: 'rail' }, marker(n, color, dashed), h('div', { class: 'line' })),
      h('div', { class: 'tbody' }, head, body));
  }

  function renderCenter(task, state) {
    var box = clear($('center'));
    var panel = h('section', { class: 'panel timeline' },
      h('div', { class: 'row-between wrap' },
        h('h2', null, 'What the agent did ', h('span', { class: 'muted small', text: 'turn by turn' })),
        h('div', { class: 'legend small' },
          h('span', null, h('i', { style: 'background:' + COLORS.think }), 'thinking'),
          h('span', null, h('i', { style: 'background:' + COLORS.python }), 'python'),
          h('span', null, h('i', { style: 'background:' + COLORS.submit }), 'submit'))));
    box.appendChild(panel);
    if (!task || !state) { panel.appendChild(h('p', { class: 'muted', text: 'Select a task.' })); return; }
    if (!state.count) { panel.appendChild(h('p', { class: 'muted', text: 'Nothing recorded for this task yet.' })); return; }
    var multiAtt = state.cards.some(function (c) { return c.att > 0; });
    state.cards.forEach(function (c, i) {
      var key = task.id + ':' + i;
      var attTxt = multiAtt ? 'att ' + c.att + ' · ' : '';
      var body, r;
      if (c.kind === 'think') {
        var ev = c.ev, tok = num(ev.completion_tokens);
        var pct = Math.max(3, Math.min(100, state.maxTokens ? (tok / state.maxTokens) * 100 : 3));
        var bar = h('div', { class: 'bar' }, h('div', { class: 'barfill' + (ev.finish === 'length' ? ' cut' : '') }));
        bar.firstChild.style.width = pct.toFixed(0) + '%';
        var full = S.expanded[key];
        var reasoning = typeof ev.reasoning === 'string' ? ev.reasoning : '';
        var finishChip = h('span', { class: 'chip small ' + (ev.finish === 'length' || ev.finish === 'error' ? 'badc' : 'neutral'),
          text: ev.finish === 'length' ? 'cut off at token cap (length)' : 'finish: ' + (ev.finish || '?') });
        body = h('div', { class: 'card think' }, bar,
          h('div', { class: 'mono small muted spaced' }, formatInt(tok) + ' completion tokens'
            + (ev.reasoning_tokens != null ? ' (' + formatInt(ev.reasoning_tokens) + ' reasoning)' : '') + ' · ', finishChip),
          full ? h('pre', { class: 'full', text: reasoning || '(no reasoning stored)' })
            : h('div', { class: 'excerpt', text: reasoning ? excerpt(reasoning) : '(no reasoning stored)' }),
          ev.content ? h('div', { class: 'reply small', text: 'Reply: ' + (full ? ev.content : excerpt(ev.content, 240)) }) : null,
          reasoning ? h('button', { class: 'btn small', type: 'button', 'aria-expanded': full ? 'true' : 'false',
            on: { click: function () { S.expanded[key] = !S.expanded[key]; render(true); } },
            text: full ? 'Hide full reasoning' : 'Show full reasoning' }) : null);
        r = row(c.turn, COLORS.think, 'THINKING', attTxt + formatDur(ev.dur_s), body);
      } else if (c.kind === 'python' || c.kind === 'tool') {
        var code = c.call.args && typeof c.call.args.code === 'string' ? c.call.args.code : JSON.stringify(c.call.args || {});
        body = h('div', { class: 'card code' }, h('pre', { class: 'src', text: code }),
          h('div', { class: 'out mono', text: c.result ? String(c.result.text == null ? '' : c.result.text) : 'running…' }));
        r = row(c.turn, COLORS.python, c.kind === 'python' ? 'PYTHON' : String(c.name || 'TOOL').toUpperCase(),
          attTxt + (c.result ? formatDur(c.result.dur_s) : 'pending'), body);
      } else if (c.kind === 'submit') {
        var s = c.submit;
        var chipText = s ? num(s.reproduced) + ' of ' + num(s.n_train) + ' reproduced' : 'evaluating…';
        var prog = s && typeof s.program === 'string' ? s.program : (c.call.args && c.call.args.code) || '';
        body = h('div', { class: 'card submitc' },
          h('div', { class: 'row-gap' },
            h('span', { class: 'chip ' + (s && s.verified ? 'good' : s ? 'badc' : 'neutral'), text: chipText }),
            h('span', { class: 'small', text: s ? submitDetail(s) : '' })),
          s && Array.isArray(s.flags) && s.flags.length ? h('div', { class: 'small flagline', text: 'flags: ' + s.flags.join('; ') }) : null,
          prog ? h('details', null, h('summary', { class: 'small', text: 'Program (' + prog.split('\n').length + ' lines)' }), h('pre', { class: 'src', text: prog })) : null,
          c.result && c.result.text ? h('details', null, h('summary', { class: 'small', text: 'Reply sent to the model' }), h('pre', { class: 'src', text: String(c.result.text) })) : null);
        r = row(c.turn, COLORS.submit, 'SUBMIT', attTxt + (c.progNo ? 'program ' + c.progNo : '') + (prog ? ' · ' + prog.split('\n').length + ' lines' : ''), body);
      } else if (c.kind === 'rescue') {
        r = row(c.turn, COLORS.rescue, 'RESCUE TURN', attTxt, h('div', { class: 'card small', text: String(c.ev.reason || '') }));
      } else {
        var e = c.ev, okEnd = e.verified === true;
        r = row('✓', okEnd ? COLORS.ok : COLORS.bad, 'ATTEMPT ' + c.att + ' ENDED', String(e.status || ''),
          h('div', { class: 'card small mono', text: formatInt(e.tokens) + ' tokens · ' + num(e.turns) + ' turns · ' + num(e.python_calls) + ' python · ' + num(e.submits) + ' submits · ' + formatDur(e.wall_s) }));
      }
      panel.appendChild(r);
    });
    if (state.thinkingNow) {
      var tn = state.thinkingNow;
      panel.appendChild(row(null, COLORS.think, 'THINKING NOW', 'turn ' + tn.turn + ' · since ' + formatClock(tn.since),
        h('div', { class: 'small muted', text: 'No reply recorded yet. The trace has no further event for this attempt.' }), true));
    }
  }

  function renderScrubber(task, state) {
    var track = clear($('ticks'));
    var evs = task ? task.events : [];
    var n = evs.length;
    $('scrub').setAttribute('aria-valuemax', String(Math.max(0, n - 1)));
    $('scrub').setAttribute('aria-valuenow', String(Math.min(S.idx, Math.max(0, n - 1))));
    $('scrub').setAttribute('aria-valuetext', n ? 'event ' + (S.idx + 1) + ' of ' + n + ', ' + evs[S.idx].ev : 'no events');
    evs.forEach(function (e, i) {
      var tk = document.createElement('div');
      tk.className = 'tick' + (i === S.idx ? ' current' : '');
      tk.style.background = tickColor(e);
      tk.style.opacity = i <= S.idx ? '1' : '0.35';
      tk.title = (i + 1) + ': ' + e.ev + (e.turn != null ? ' (turn ' + e.turn + ')' : '');
      tk.addEventListener('click', function () { seek(i); });
      track.appendChild(tk);
    });
    var cur = n ? evs[Math.min(S.idx, n - 1)] : null;
    $('s-left').textContent = n ? formatClock(evs[0].t) : '00:00:00';
    $('s-mid').textContent = n ? 'event ' + (S.idx + 1) + ' of ' + n + (state && state.lastTurn ? ' · turn ' + state.lastTurn.turn : '') + ' · ' + formatClock(cur.t) : '--';
    $('s-right').textContent = n ? formatClock(evs[n - 1].t) : '--';
    $('btn-play').textContent = S.playing ? 'Pause' : 'Play';
    $('btn-play').setAttribute('aria-label', S.playing ? 'Pause replay' : 'Play replay');
    ['btn-prev', 'btn-next', 'btn-play'].forEach(function (id) { $(id).disabled = !n; });
    var sp = clear($('speeds'));
    api.SPEEDS.forEach(function (v) {
      sp.appendChild(h('button', { class: 'btn speed', type: 'button', 'aria-pressed': v === S.speed ? 'true' : 'false',
        'aria-label': 'Playback speed ' + v + ' times', on: { click: function () { S.speed = v; render(true); } }, text: v + '×' }));
    });
  }

  function render(keepScroll) {
    var task = curTask();
    var n = task ? task.events.length : 0;
    if (S.idx > n - 1) S.idx = Math.max(0, n - 1);
    var state = task && n ? stateAt(task, S.idx) : null;
    var main = $('main');
    var scrollTop = main ? main.scrollTop : 0;
    $('empty').hidden = !!S.run;
    $('layout').hidden = !S.run;
    renderHeader();
    renderRight(state);
    renderLeft(task, state);
    renderCenter(task, state);
    renderScrubber(task, state);
    if (keepScroll && main) main.scrollTop = scrollTop;
  }

  function selectTask(id) {
    stop();
    S.taskId = id; S.tab = 0;
    var t = curTask();
    S.idx = t && t.events.length ? t.events.length - 1 : 0;
    render();
  }
  function seek(i) {
    var t = curTask();
    if (!t || !t.events.length) return;
    S.idx = Math.max(0, Math.min(t.events.length - 1, i));
    render(true);
  }
  function stop() { S.playing = false; if (S.timer) { clearTimeout(S.timer); S.timer = null; } }
  function schedule() {
    var t = curTask();
    if (!S.playing || !t) return;
    if (S.idx >= t.events.length - 1) { stop(); render(true); return; }
    var ms = playbackDelayMs(t.events[S.idx].t, t.events[S.idx + 1].t, S.speed);
    S.timer = setTimeout(function () {
      S.idx++;
      render(true);
      var last = document.querySelector('#center .trow:last-child');
      if (last && last.scrollIntoView) last.scrollIntoView({ block: 'nearest' });
      schedule();
    }, ms);
  }
  function togglePlay() {
    var t = curTask();
    if (!t || !t.events.length) return;
    if (S.playing) { stop(); render(true); return; }
    if (S.idx >= t.events.length - 1) S.idx = 0;
    S.playing = true;
    render(true);
    schedule();
  }

  function loadText(text, name) {
    stop();
    var p = parseJsonl(text);
    S.events = p.events;
    var cfgs = listConfigs(p.events);
    S.cfg = cfgs.indexOf(S.cfg) >= 0 ? S.cfg : cfgs[0] || null;
    S.notes = name + ': ' + p.events.length + ' events'
      + (p.truncated ? ', truncated last line ignored' : '') + (p.bad ? ', ' + p.bad + ' bad lines skipped' : '')
      + (p.unknown ? ', ' + p.unknown + ' unknown events ignored' : '');
    S.run = p.events.length ? buildRun(p.events, S.cfg) : null;
    if (!S.run) S.notes = name + ': no usable events found';
    if (S.run) {
      var pick = S.run.tasks.filter(function (t) { return t.id === S.taskId; })[0]
        || S.run.tasks.filter(function (t) { return taskStatus(t.events) === 'running'; })[0]
        || S.run.tasks.filter(function (t) { return t.events.length; })[0] || S.run.tasks[0];
      S.taskId = pick ? pick.id : null;
      var t = curTask();
      S.idx = t && t.events.length ? t.events.length - 1 : 0;
      S.tab = 0; S.expanded = {};
    }
    render();
  }
  function loadFile(file) {
    if (!file) return;
    S.file = file;
    file.text().then(function (txt) { loadText(txt, file.name); }, function (err) {
      S.notes = 'Could not read ' + file.name + ': ' + (err && err.message ? err.message : 'error') + '. Pick the file again.';
      render();
    });
  }

  function onKey(ev) {
    var tg = ev.target, tag = tg && tg.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (ev.key === 'ArrowLeft') { ev.preventDefault(); stop(); seek(S.idx - 1); }
    else if (ev.key === 'ArrowRight') { ev.preventDefault(); stop(); seek(S.idx + 1); }
    else if (ev.key === 'Home') { ev.preventDefault(); stop(); seek(0); }
    else if (ev.key === 'End') { ev.preventDefault(); stop(); seek(1e9); }
    else if (ev.key === ' ' && tag !== 'BUTTON' && tag !== 'SUMMARY') { ev.preventDefault(); togglePlay(); }
  }

  function init() {
    $('file').addEventListener('change', function (e) { loadFile(e.target.files && e.target.files[0]); e.target.value = ''; });
    $('btn-open').addEventListener('click', function () { $('file').click(); });
    $('btn-reload').addEventListener('click', function () { if (S.file) loadFile(S.file); else $('file').click(); });
    $('cfgsel').addEventListener('change', function (e) {
      stop(); S.cfg = e.target.value; S.run = buildRun(S.events, S.cfg);
      var first = S.run.tasks.filter(function (t) { return t.events.length; })[0] || S.run.tasks[0];
      S.taskId = first ? first.id : null; selectTask(S.taskId);
    });
    $('btn-prev').addEventListener('click', function () { stop(); seek(S.idx - 1); });
    $('btn-next').addEventListener('click', function () { stop(); seek(S.idx + 1); });
    $('btn-play').addEventListener('click', togglePlay);
    document.addEventListener('keydown', onKey);
    document.addEventListener('dragover', function (e) { e.preventDefault(); document.body.classList.add('dragging'); });
    document.addEventListener('dragleave', function (e) { if (!e.relatedTarget) document.body.classList.remove('dragging'); });
    document.addEventListener('drop', function (e) {
      e.preventDefault(); document.body.classList.remove('dragging');
      loadFile(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]);
    });
    render();
  }
  init();
})();
