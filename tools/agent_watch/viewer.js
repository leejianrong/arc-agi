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
      return 'Example ' + (num(x.ex) + 1) + ': ' + plural(num(x.total), 'cell') + ' wrong.';
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

  /* '1 cell', '2 cells', '0 cells'. */
  function plural(n, one, many) {
    var v = num(n);
    return v === 1 ? '1 ' + one : formatInt(v) + ' ' + (many || one + 's');
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


  /* ---------------------------------------------------------------- run wall */

  function sumTokens(run) {
    var tokens = 0;
    run.tasks.forEach(function (t) {
      t.events.forEach(function (e) { if (e.ev === 'turn') tokens += num(e.completion_tokens); });
    });
    return tokens;
  }

  /* One comparison row for a config's run: run_end.summary fields win, anything
   * missing is computed from events. solved is null (unscored, never 0) when no
   * task_end carries solved data and the summary has no count. */
  function configSummary(run) {
    var tot = computeTotals(run);
    var hit = 0, of = 0, scored = false;
    run.tasks.forEach(function (t) {
      t.events.forEach(function (e) {
        if (e.ev !== 'task_end' || !Array.isArray(e.solved)) return;
        e.solved.forEach(function (v) {
          if (typeof v !== 'boolean') return;
          scored = true; of++;
          if (v) hit++;
        });
      });
    });
    var sm = run.end && run.end.summary && typeof run.end.summary === 'object' ? run.end.summary : null;
    function pick(key, computed) { return sm && typeof sm[key] === 'number' && isFinite(sm[key]) ? sm[key] : computed; }
    var wall = pick('wall_s', run.lastT);
    var tokens = pick('tokens', tot.tokens);
    var solved = pick('solved_test_inputs', scored ? hit : null);
    return {
      name: run.cfg == null ? 'default' : run.cfg, source: sm ? 'run_end' : 'events', hasTrace: true,
      tasks: pick('tasks', run.tasks.length), verified: pick('verified', tot.verified), tokens: tokens,
      tokS: pick('tok_s', wall > 0 ? tokens / wall : null), wallS: wall,
      solved: solved, solvedOf: scored ? of : null, unscored: solved == null, error: null
    };
  }

  function numOrNull(v) { return typeof v === 'number' && isFinite(v) ? v : null; }

  /* A bench_results.json row (see kaggle/arc2_synth/launcher.py.tmpl): name, mode,
   * tasks, wall_s, tokens, tok_s, verified (may be null), score ([hit, total] or
   * null). A plain `solved` number or "h/t" string is accepted too. */
  function normalizeBenchRow(r) {
    var solved = null, of = null, m;
    if (Array.isArray(r.score) && numOrNull(r.score[0]) != null && numOrNull(r.score[1]) != null) { solved = r.score[0]; of = r.score[1]; }
    else if (numOrNull(r.solved) != null) solved = r.solved;
    else if (typeof r.solved === 'string' && (m = /^\s*(\d+)\s*\/\s*(\d+)\s*$/.exec(r.solved))) { solved = +m[1]; of = +m[2]; }
    return {
      name: r.name, mode: typeof r.mode === 'string' ? r.mode : null, source: 'bench', hasTrace: false,
      tasks: numOrNull(r.tasks), verified: numOrNull(r.verified), tokens: numOrNull(r.tokens), tokS: numOrNull(r.tok_s),
      wallS: numOrNull(r.wall_s), solved: solved, solvedOf: of, unscored: solved == null,
      error: typeof r.error === 'string' ? r.error : null
    };
  }

  /* Rows of a bench_results.json text (a JSON array, or an object holding a list
   * under rows/results/configs), or null when the text is not that. */
  function parseBench(text) {
    var o;
    try { o = JSON.parse(String(text)); } catch (err) { return null; }
    var list = null;
    if (Array.isArray(o)) list = o;
    else if (o && typeof o === 'object' && typeof o.ev !== 'string') {
      ['rows', 'results', 'configs'].forEach(function (k) { if (!list && Array.isArray(o[k])) list = o[k]; });
    }
    if (!list) return null;
    var rows = list.filter(function (r) { return r && typeof r === 'object' && !Array.isArray(r) && typeof r.name === 'string'; })
      .map(normalizeBenchRow);
    return rows.length ? rows : null;
  }

  /* Comparison table rows: every loaded trace config, then bench rows whose name
   * has no trace. `runs` is [{cfg, run}]. */
  function comparisonRows(runs, bench) {
    var rows = runs.map(function (r) { return configSummary(r.run); });
    var have = {};
    rows.forEach(function (r) { have[r.name] = true; });
    (bench || []).forEach(function (b) { if (!have[b.name]) { have[b.name] = true; rows.push(b); } });
    return rows;
  }

  /* S solved, V verified but not solved (or solved unknown), F failed, '-' not
   * reached (nothing recorded, or still running). */
  function taskOutcome(task) {
    if (!task || !task.events.length) return '-';
    var end = null;
    task.events.forEach(function (e) { if (e.ev === 'task_end') end = e; });
    if (end) {
      var sol = Array.isArray(end.solved) ? end.solved.filter(function (v) { return typeof v === 'boolean'; }) : [];
      if (sol.length && sol.every(function (v) { return v; })) return 'S';
      return num(end.verified) > 0 || taskStatus(task.events) === 'verified' ? 'V' : 'F';
    }
    var st = taskStatus(task.events);
    return st === 'verified' ? 'V' : st === 'unverified' ? 'F' : '-';
  }

  function outcomeMatrix(runs) {
    var ids = [], seen = {};
    runs.forEach(function (r) {
      r.run.tasks.forEach(function (t) { if (!seen[t.id]) { seen[t.id] = true; ids.push(t.id); } });
    });
    var maps = runs.map(function (r) {
      var m = {};
      r.run.tasks.forEach(function (t) { m[t.id] = t; });
      return m;
    });
    return {
      cfgs: runs.map(function (r) { return r.cfg; }),
      rows: ids.map(function (id) { return { id: id, outcomes: maps.map(function (m) { return taskOutcome(m[id]); }) }; })
    };
  }

  /* What a wall tile shows: the final attempt_1 when there is one, else the first
   * train input. */
  function tileInfo(task) {
    var turns = 0, tokens = 0, final = null, grid = null, gridKind = null;
    task.events.forEach(function (e) {
      if (e.ev === 'turn') { turns++; tokens += num(e.completion_tokens); }
      if (e.ev === 'task_end' && Array.isArray(e.rows) && e.rows[0] && gridDims(e.rows[0].attempt_1)[0]) final = e.rows[0].attempt_1;
    });
    if (final) { grid = final; gridKind = 'attempt'; }
    else if (task.start && Array.isArray(task.start.train) && task.start.train[0] && gridDims(task.start.train[0].input)[0]) {
      grid = task.start.train[0].input; gridKind = 'input';
    }
    return { id: task.id, status: taskStatus(task.events), turns: turns, tokens: tokens, grid: grid, gridKind: gridKind };
  }

  var api = {
    PALETTE: PALETTE, SPEEDS: [1, 2, 4, 8, 32], parseJsonl: parseJsonl, listConfigs: listConfigs, buildRun: buildRun,
    taskStatus: taskStatus, stateAt: stateAt, gridDiff: gridDiff, countPairs: countPairs,
    pickSubmitExample: pickSubmitExample, submitDetail: submitDetail, excerpt: excerpt, formatClock: formatClock,
    formatInt: formatInt, plural: plural, formatDur: formatDur, computeTotals: computeTotals, heartbeatSeries: heartbeatSeries,
    tickColor: tickColor, playbackDelayMs: playbackDelayMs, escapeHtml: escapeHtml,
    configSummary: configSummary, normalizeBenchRow: normalizeBenchRow, parseBench: parseBench,
    comparisonRows: comparisonRows, taskOutcome: taskOutcome, outcomeMatrix: outcomeMatrix, tileInfo: tileInfo
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof document === 'undefined') return;

  /* ------------------------------------------------------------------ DOM */

  var S = {
    events: [], cfg: null, run: null, taskId: null, idx: 0, speed: 8, playing: false, timer: null,
    tab: 0, expanded: {}, files: [], notes: '', view: 'watch', bench: [], wallRuns: [], wallCfg: null
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
    sel.hidden = cfgs.length < 2 || S.view !== 'watch';
    cfgs.forEach(function (c) {
      var o = h('option', { value: c, text: c });
      if (c === S.cfg) o.selected = true;
      sel.appendChild(o);
    });
    var wallTab = $('tab-wall'), watchTab = $('tab-watch');
    wallTab.setAttribute('aria-pressed', S.view === 'wall' ? 'true' : 'false');
    watchTab.setAttribute('aria-pressed', S.view === 'watch' ? 'true' : 'false');
    if (S.view === 'wall') {
      chip.textContent = S.wallRuns.length + (S.wallRuns.length === 1 ? ' config' : ' configs') + ' from traces'
        + (S.bench.length ? ', ' + S.bench.length + ' from bench_results' : '');
    }
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
      if (cells.length > 10) lines.appendChild(h('div', { class: 'muted', text: '+' + (cells.length - 10) + ' more wrong ' + (cells.length - 10 === 1 ? 'cell' : 'cells') }));
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


  /* ------------------------------------------------------------------ run wall */

  var CFG_COLORS = ['#3DD9C0', '#8E7CF8', '#6CB6FF', '#FFC857', '#FF851B', '#F012BE', '#2ECC40', '#7FDBFF'];
  var OUTCOME = {
    S: { label: 'solved', bg: '#16301F', fg: '#4ADE80' },
    V: { label: 'verified, not solved', bg: '#3A2F18', fg: '#FFC857' },
    F: { label: 'failed', bg: '#3A2024', fg: '#FF8A8A' },
    '-': { label: 'not reached', bg: '#232A33', fg: '#9AA4B2' }
  };
  var TILE_RING = { verified: '#2F5A3F', running: '#6B5420', unverified: '#6B2A31', queued: '#2A313B' };

  function wallRun(cfg) {
    for (var i = 0; i < S.wallRuns.length; i++) if (S.wallRuns[i].cfg === cfg) return S.wallRuns[i].run;
    return null;
  }

  function openTask(cfg, id) {
    stop();
    S.view = 'watch';
    if (cfg != null) { S.cfg = cfg; S.run = buildRun(S.events, cfg); }
    selectTask(id);
  }

  function fmtOpt(v, f) { return v == null ? '--' : f(v); }

  function renderWall() {
    var box = clear($('wall'));
    var runs = S.wallRuns;
    if (!runs.some(function (r) { return r.cfg === S.wallCfg; })) S.wallCfg = runs.length ? runs[0].cfg : null;

    var rows = comparisonRows(runs, S.bench);
    var tbody = h('tbody');
    rows.forEach(function (r, i) {
      var solvedTxt = r.unscored ? 'unscored' : String(r.solved) + (r.solvedOf != null ? ' / ' + r.solvedOf : '');
      var swatch = h('span', { class: 'swatch' });
      swatch.style.background = CFG_COLORS[i % CFG_COLORS.length];
      var nameCell = h('td', { class: 'left' }, swatch, h('span', { text: r.name }));
      if (r.error) nameCell.appendChild(h('span', { class: 'chip small badc', title: r.error, text: 'error' }));
      var src = r.source === 'run_end' ? 'trace, run_end summary' : r.source === 'events' ? 'trace, computed from events' : 'bench_results.json';
      tbody.appendChild(h('tr', { title: src },
        nameCell,
        h('td', { text: fmtOpt(r.tasks, formatInt) }),
        h('td', { text: fmtOpt(r.tokens, formatInt) }),
        h('td', { text: fmtOpt(r.tokS, function (v) { return formatInt(v); }) }),
        h('td', { text: fmtOpt(r.verified, formatInt) }),
        h('td', { class: 'solved' + (r.unscored ? ' unscored' : ''), text: solvedTxt }),
        h('td', { text: fmtOpt(r.wallS, formatDur) }),
        h('td', { class: 'srccell muted', text: r.source === 'bench' ? 'bench' : 'trace' })));
    });
    function th(t, left) { return h('th', { class: left ? 'left' : null, text: t }); }
    var compare = h('section', { class: 'panel wide' },
      h('h2', { text: 'Config comparison' }),
      h('div', { class: 'muted small', text: 'Summary from run_end when the trace has one, otherwise computed from its events. Solved needs public answers: without them it reads unscored.' }),
      h('div', { class: 'tablewrap' }, h('table', { class: 'cmp mono' },
        h('thead', null, h('tr', null, th('CONFIG', true), th('TASKS'), th('TOKENS'), th('TOK/S'), th('VERIFIED'), th('SOLVED'), th('WALL'), th('FROM'))),
        tbody)));
    box.appendChild(compare);

    var mx = outcomeMatrix(runs);
    var matrix = h('section', { class: 'panel mid' },
      h('h2', { text: 'Per-task outcome by config' }),
      h('div', { class: 'muted small', text: 'S solved · V verified but not solved · F failed · - not reached. Click a letter to open that task in the Watch view.' }));
    if (!mx.rows.length) {
      matrix.appendChild(h('p', { class: 'muted small', text: runs.length ? 'No tasks listed in the loaded traces.' : 'No trace loaded: only the table has data.' }));
    } else {
      var cols = 'minmax(110px, 1fr) repeat(' + mx.cfgs.length + ', minmax(64px, 96px))';
      var grid = h('div', { class: 'matrix' });
      var head = h('div', { class: 'mrow mhead' }, h('span', { class: 'grow', text: 'TASK' }));
      head.style.gridTemplateColumns = cols;
      mx.cfgs.forEach(function (c) { head.appendChild(h('span', { class: 'mcol', title: c, text: c })); });
      grid.appendChild(head);
      mx.rows.forEach(function (row, i) {
        var r = h('div', { class: 'mrow' + (i % 2 ? '' : ' alt') });
        r.style.gridTemplateColumns = cols;
        r.appendChild(h('button', { class: 'mid mono', type: 'button', title: 'Open ' + row.id + ' in the Watch view',
          on: { click: function () { openTask(S.wallCfg, row.id); } }, text: row.id }));
        row.outcomes.forEach(function (o, k) {
          var info = OUTCOME[o], cfg = mx.cfgs[k];
          var b = h('button', { class: 'oc mono', type: 'button', title: row.id + ' in ' + cfg + ': ' + info.label,
            'aria-label': row.id + ' in ' + cfg + ': ' + info.label, on: { click: function () { openTask(cfg, row.id); } }, text: o });
          b.style.background = info.bg; b.style.color = info.fg;
          r.appendChild(b);
        });
        grid.appendChild(r);
      });
      matrix.appendChild(h('div', { class: 'tablewrap' }, grid));
    }
    box.appendChild(matrix);

    var tiles = h('section', { class: 'panel full' });
    var run = wallRun(S.wallCfg);
    var sel = null;
    if (runs.length > 1) {
      sel = h('select', { class: 'btn', 'aria-label': 'Config for the tile wall', on: { change: function (e) { S.wallCfg = e.target.value; renderWall(); } } });
      runs.forEach(function (r) { sel.appendChild(h('option', { value: r.cfg, text: r.cfg, selected: r.cfg === S.wallCfg })); });
    }
    tiles.appendChild(h('div', { class: 'row-between wrap' },
      h('h2', null, run ? 'Tasks in ' : 'Tasks', run ? h('span', { class: 'mono', text: S.wallCfg }) : null,
        h('span', { class: 'muted small', text: run ? '  click a tile to open it in the Watch view' : '' })),
      sel,
      h('div', { class: 'legend small' }, ['verified', 'running', 'unverified', 'queued'].map(function (k) {
        var d = h('i', { class: 'round' });
        d.style.background = STATUS[k][0];
        return h('span', null, d, k);
      }))));
    if (!run) {
      tiles.appendChild(h('p', { class: 'muted small', text: 'Load a trace.jsonl to see its tasks as tiles.' }));
    } else {
      var wallGrid = h('div', { class: 'tiles' });
      run.tasks.forEach(function (t) {
        var ti = tileInfo(t), info = STATUS[ti.status];
        var thumb = ti.grid ? gridEl(ti.grid, { box: 64, max: 8, label: ti.gridKind === 'attempt' ? 'final attempt for ' + t.id : 'first example input of ' + t.id })
          : h('div', { class: 'thumb-missing small muted', text: 'no grid' });
        var meta = ti.turns || ti.tokens ? plural(ti.turns, 'turn') + ' · ' + formatInt(ti.tokens) + ' tok' : 'not started';
        var tile = h('button', { class: 'tile', type: 'button', on: { click: function () { openTask(S.wallCfg, t.id); } } },
          h('div', { class: 'thumb' }, thumb),
          h('div', { class: 'tmeta' },
            h('div', { class: 'mono', text: t.id }),
            h('div', { class: 'tstatus small' }, h('i', { class: 'dot', style: 'background:' + info[0] }), info[1]),
            h('div', { class: 'mono muted small', text: meta })));
        tile.style.borderColor = TILE_RING[ti.status];
        wallGrid.appendChild(tile);
      });
      tiles.appendChild(wallGrid);
    }
    box.appendChild(tiles);
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
    var watch = S.view === 'watch';
    var has = !!S.run || S.bench.length > 0;
    $('empty').hidden = watch ? !!S.run : has;
    $('layout').hidden = !watch || !S.run;
    $('wall').hidden = watch || !has;
    $('footer').hidden = !watch;
    renderHeader();
    if (watch) {
      renderRight(state);
      renderLeft(task, state);
      renderCenter(task, state);
      renderScrubber(task, state);
    } else {
      renderWall();
    }
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

  /* items: [{name, text}] from one pick or drop. Traces are merged per cfg (a later
   * file replaces an earlier one with the same cfg); bench_results.json rows go to
   * the comparison table. A new load replaces what was loaded before. */
  function loadItems(items) {
    stop();
    var byCfg = {}, order = [], notes = [], bench = [];
    items.forEach(function (it) {
      var rows = parseBench(it.text);
      if (rows) { bench = bench.concat(rows); notes.push(it.name + ': ' + plural(rows.length, 'bench row')); return; }
      var p = parseJsonl(it.text);
      p.events.forEach(function (e) {
        var c = cfgOf(e);
        if (!byCfg[c] || byCfg[c].file !== it.name) { if (!byCfg[c]) order.push(c); byCfg[c] = { file: it.name, events: [] }; }
        byCfg[c].events.push(e);
      });
      notes.push(it.name + ': ' + p.events.length + ' events'
        + (p.truncated ? ', truncated last line ignored' : '') + (p.bad ? ', ' + p.bad + ' bad lines skipped' : '')
        + (p.unknown ? ', ' + p.unknown + ' unknown events ignored' : '')
        + (!p.events.length ? ' (no usable events)' : ''));
    });
    S.events = [];
    order.forEach(function (c) { S.events = S.events.concat(byCfg[c].events); });
    S.bench = bench;
    S.notes = notes.join(' | ');
    var cfgs = listConfigs(S.events);
    S.wallRuns = cfgs.map(function (c) { return { cfg: c, run: buildRun(S.events, c) }; });
    S.cfg = cfgs.indexOf(S.cfg) >= 0 ? S.cfg : cfgs[0] || null;
    S.run = S.events.length ? buildRun(S.events, S.cfg) : null;
    if (S.run) {
      var pick = S.run.tasks.filter(function (t) { return t.id === S.taskId; })[0]
        || S.run.tasks.filter(function (t) { return taskStatus(t.events) === 'running'; })[0]
        || S.run.tasks.filter(function (t) { return t.events.length; })[0] || S.run.tasks[0];
      S.taskId = pick ? pick.id : null;
      var t = curTask();
      S.idx = t && t.events.length ? t.events.length - 1 : 0;
      S.tab = 0; S.expanded = {};
    } else if (bench.length) S.view = 'wall';
    else S.notes = (S.notes ? S.notes + ' | ' : '') + 'nothing usable found';
    render();
  }
  function loadFiles(list) {
    var files = Array.prototype.slice.call(list || []);
    if (!files.length) return;
    S.files = files;
    Promise.all(files.map(function (f) { return f.text().then(function (txt) { return { name: f.name, text: txt }; }); }))
      .then(loadItems, function (err) {
        S.notes = 'Could not read the file(s): ' + (err && err.message ? err.message : 'error') + '. Pick them again.';
        render();
      });
  }

  function onKey(ev) {
    var tg = ev.target, tag = tg && tg.tagName;
    if (S.view !== 'watch') return;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (ev.key === 'ArrowLeft') { ev.preventDefault(); stop(); seek(S.idx - 1); }
    else if (ev.key === 'ArrowRight') { ev.preventDefault(); stop(); seek(S.idx + 1); }
    else if (ev.key === 'Home') { ev.preventDefault(); stop(); seek(0); }
    else if (ev.key === 'End') { ev.preventDefault(); stop(); seek(1e9); }
    else if (ev.key === ' ' && tag !== 'BUTTON' && tag !== 'SUMMARY') { ev.preventDefault(); togglePlay(); }
  }

  function init() {
    $('file').addEventListener('change', function (e) { loadFiles(e.target.files); e.target.value = ''; });
    $('tab-watch').addEventListener('click', function () { stop(); S.view = 'watch'; render(); });
    $('tab-wall').addEventListener('click', function () { stop(); S.view = 'wall'; render(); });
    $('btn-open').addEventListener('click', function () { $('file').click(); });
    $('btn-reload').addEventListener('click', function () { if (S.files.length) loadFiles(S.files); else $('file').click(); });
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
      loadFiles(e.dataTransfer && e.dataTransfer.files);
    });
    render();
  }
  init();
})();
