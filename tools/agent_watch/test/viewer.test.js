'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const V = require('../viewer.js');

const sample = fs.readFileSync(path.join(__dirname, '..', 'sample', 'sample_trace.jsonl'), 'utf8');
const parsed = V.parseJsonl(sample);
const run = V.buildRun(parsed.events, 'default');
const byId = (id) => run.tasks.find((t) => t.id === id);

test('parseJsonl ignores a truncated last line, blanks and unknown events', () => {
  const text = '{"ev":"run_start","t":0}\n\n{"ev":"mystery","t":1}\nnot json\n{"ev":"turn","t":2,"task":"a"}\n{"ev":"turn","t":3,"ta';
  const p = V.parseJsonl(text);
  assert.equal(p.events.length, 2);
  assert.equal(p.truncated, true);
  assert.equal(p.bad, 1);
  assert.equal(p.unknown, 1);
  assert.equal(V.parseJsonl('{"ev":"run_end","t":1}\n').truncated, false);
});

test('unknown fields are tolerated', () => {
  const p = V.parseJsonl('{"ev":"heartbeat","t":1,"tok_s":5,"future":{"x":1}}');
  assert.equal(p.events[0].tok_s, 5);
});

test('sample trace parses cleanly and groups by task', () => {
  assert.equal(parsed.truncated, false);
  assert.equal(parsed.bad, 0);
  assert.deepEqual(run.tasks.map((t) => t.id), ['135a2760', '136b0064', '142ca369', '16b78196', '1818057f']);
  assert.ok(byId('135a2760').events.every((e) => e.task === '135a2760'));
  assert.equal(run.heartbeats.length, 4);
  assert.equal(run.end, null);
});

test('listConfigs and per-config runs', () => {
  const evs = [{ ev: 'run_start', cfg: 'a', t: 0, tasks: ['x'] }, { ev: 'run_start', cfg: 'b', t: 0, tasks: ['y'] }, { ev: 'heartbeat', t: 4 }];
  assert.deepEqual(V.listConfigs(evs), ['a', 'b', 'default']);
  assert.deepEqual(V.buildRun(evs, 'b').tasks.map((t) => t.id), ['y']);
});

test('task status derivation', () => {
  assert.equal(V.taskStatus(byId('135a2760').events), 'verified');
  assert.equal(V.taskStatus(byId('136b0064').events), 'unverified');
  assert.equal(V.taskStatus(byId('142ca369').events), 'running');
  assert.equal(V.taskStatus(byId('16b78196').events), 'queued');
  assert.equal(V.taskStatus([{ ev: 'task_start', task: 'z', t: 0 }]), 'running');
});

test('totals: elapsed from last t, tokens from turns, tok/s from the last heartbeat', () => {
  const tot = V.computeTotals(run);
  const last = Math.max(...parsed.events.map((e) => e.t));
  assert.equal(tot.elapsed, last);
  const sum = parsed.events.filter((e) => e.ev === 'turn').reduce((a, e) => a + e.completion_tokens, 0);
  assert.equal(tot.tokens, sum);
  assert.equal(tot.tokS, 259);
  assert.equal(tot.verified, 1);
  assert.equal(V.computeTotals({ tasks: [], heartbeats: [], lastT: 0 }).tokS, null);
});

test('stateAt builds ordered cards and matches results and submits', () => {
  const st = V.stateAt(byId('135a2760'), null);
  const kinds = st.cards.map((c) => c.kind);
  assert.deepEqual(kinds, ['think', 'python', 'think', 'submit', 'think', 'submit', 'think', 'end']);
  assert.match(st.cards[1].result.text, /cells differ/);
  assert.equal(st.cards[3].submit.reproduced, 1);
  assert.equal(st.cards[3].progNo, 1);
  assert.equal(st.cards[5].progNo, 2);
  assert.equal(st.cards[5].submit.verified, true);
  assert.equal(st.status, 'verified');
  assert.equal(st.thinkingNow, null);
});

test('stateAt truncates at an event index (scrubber)', () => {
  const task = byId('135a2760');
  const idxFirstSubmit = task.events.findIndex((e) => e.ev === 'submit');
  const st = V.stateAt(task, idxFirstSubmit);
  assert.equal(st.latestSubmit.reproduced, 1);
  assert.equal(st.status, 'running');
  assert.equal(st.cards.filter((c) => c.kind === 'think').length, 2);
  const before = V.stateAt(task, 0);
  assert.equal(before.cards.length, 0);
  assert.equal(before.latestSubmit, null);
  assert.deepEqual(before.thinkingNow && before.thinkingNow.turn, 1);
  assert.equal(V.stateAt(task, 999).count, task.events.length);
  assert.equal(V.stateAt(task, -1).count, 0);
});

test('unfinished tasks get a thinking-now state; cut-off turn is kept', () => {
  const st = V.stateAt(byId('142ca369'), null);
  assert.equal(st.status, 'running');
  assert.equal(st.thinkingNow.turn, 3);
  assert.equal(st.lastTurn.finish, 'length');
  assert.equal(V.stateAt(byId('16b78196'), null).count, 0);
});

test('a pending tool call means not thinking', () => {
  const task = { events: [
    { ev: 'task_start', task: 'q', t: 0 },
    { ev: 'turn', task: 'q', att: 0, turn: 1, t: 1, completion_tokens: 5, tool_calls: [{ id: 'a', name: 'python', args: { code: 'x' } }] }] };
  assert.equal(V.stateAt(task, null).thinkingNow, null);
});

test('gridDiff lists wrong cells and handles shape mismatch', () => {
  const d = V.gridDiff([[1, 2], [3, 4]], [[1, 5], [3, 6]]);
  assert.deepEqual(d.cells, [[0, 1, 5, 2], [1, 1, 6, 4]]);
  assert.equal(d.mask['0,1'], true);
  assert.equal(d.shapeMismatch, false);
  const m = V.gridDiff([[1, 2]], [[1]]);
  assert.equal(m.shapeMismatch, true);
  assert.deepEqual(m.cells, [[0, 1, null, 2]]);
  assert.deepEqual(V.gridDiff([[1]], null).cells, []);
  assert.deepEqual(V.countPairs([[0, 0, 6, 5], [0, 1, 6, 5], [1, 1, 5, 6]]).map((p) => p.n), [2, 1]);
});

test('submit helpers', () => {
  const sub = parsed.events.find((e) => e.ev === 'submit' && e.verified === false);
  assert.equal(V.pickSubmitExample(sub), 1);
  assert.match(V.submitDetail(sub), /Example 2: 1 cell wrong/);
  assert.equal(V.pickSubmitExample(null), 0);
});

test('palette is the standard ARC palette', () => {
  assert.equal(V.PALETTE.length, 10);
  assert.equal(V.PALETTE[0], '#000000');
  assert.equal(V.PALETTE[9], '#870C25');
});

test('tick colours differ by type', () => {
  const set = new Set(['turn', 'submit', 'rescue'].map((ev) => V.tickColor({ ev })));
  assert.equal(set.size, 3);
  assert.notEqual(V.tickColor({ ev: 'tool_result', name: 'python' }), V.tickColor({ ev: 'turn' }));
  assert.notEqual(V.tickColor({ ev: 'attempt_end', verified: true }), V.tickColor({ ev: 'attempt_end', verified: false }));
});

test('playback delay scales with the t gap, divided by speed, and caps at 1.5 s', () => {
  assert.equal(V.playbackDelayMs(10, 12, 1), 1500);
  assert.equal(V.playbackDelayMs(10, 11, 2), 500);
  assert.equal(V.playbackDelayMs(10, 11, 8), 125);
  assert.equal(V.playbackDelayMs(0, 600, 1), 1500);
  assert.equal(V.playbackDelayMs(0, 600, 32), 1500);
  assert.equal(V.playbackDelayMs(5, 5, 1), 30);
  assert.equal(V.playbackDelayMs(5, 4, 1), 30);
});

test('formatting helpers', () => {
  assert.equal(V.formatClock(2467), '00:41:07');
  assert.equal(V.formatInt(612480), '612,480');
  assert.equal(V.excerpt('a  b\n c', 100), 'a b c');
  assert.equal(V.excerpt('x'.repeat(10), 4), 'xxxx…');
});

test('escapeHtml neutralises markup', () => {
  assert.equal(V.escapeHtml('<img src=x onerror="a()">&\''), '&lt;img src=x onerror=&quot;a()&quot;&gt;&amp;&#39;');
});

/* ------------------------------------------------------------ run wall */

const sampleB = fs.readFileSync(path.join(__dirname, '..', 'sample', 'sample_trace_b.jsonl'), 'utf8');
const eventsAll = V.parseJsonl(sample + sampleB).events;
const runsAB = V.listConfigs(eventsAll).map((c) => ({ cfg: c, run: V.buildRun(eventsAll, c) }));

test('plural helper', () => {
  assert.equal(V.plural(0, 'cell'), '0 cells');
  assert.equal(V.plural(1, 'cell'), '1 cell');
  assert.equal(V.plural(2, 'cell'), '2 cells');
  assert.equal(V.plural(1234, 'cell'), '1,234 cells');
  assert.equal(V.plural(2, 'bench row'), '2 bench rows');
  assert.equal(V.plural(3, 'child', 'children'), '3 children');
  assert.equal(V.submitDetail({ verified: false, wrong: [{ ex: 0, total: 1 }, { ex: 1, total: 2 }] }),
    'Example 1: 1 cell wrong. Example 2: 2 cells wrong.');
});

test('two sample configs are listed', () => {
  assert.deepEqual(V.listConfigs(eventsAll), ['default', 'fresh']);
});

test('configSummary is computed from events when run_end is absent', () => {
  const s = V.configSummary(runsAB[0].run);
  assert.equal(s.name, 'default');
  assert.equal(s.source, 'events');
  assert.equal(s.tasks, 5);
  assert.equal(s.verified, 1);
  assert.equal(s.tokens, 6400 + 900 + 5200 + 160 + 3100 + 2400 + 4100 + 4800 + 49152);
  assert.equal(s.wallS, runsAB[0].run.lastT);
  assert.ok(Math.abs(s.tokS - s.tokens / s.wallS) < 1e-9);
  assert.equal(s.solved, 1);
  assert.equal(s.solvedOf, 2);
  assert.equal(s.unscored, false);
});

test('configSummary prefers the run_end summary', () => {
  const s = V.configSummary(runsAB[1].run);
  assert.equal(s.source, 'run_end');
  assert.equal(s.tokens, 6500);
  assert.equal(s.tokS, 105);
  assert.equal(s.wallS, 62);
  assert.equal(s.verified, 2);
  assert.equal(s.solved, 1);
});

test('configSummary reads unscored, never 0, without solved data', () => {
  const evs = V.parseJsonl([
    '{"ev":"run_start","t":0,"cfg":"x","tasks":["a"]}',
    '{"ev":"turn","t":5,"cfg":"x","task":"a","att":0,"turn":1,"completion_tokens":50}',
    '{"ev":"task_end","t":10,"cfg":"x","task":"a","rows":[],"verified":1,"solved":null}'].join('\n')).events;
  const s = V.configSummary(V.buildRun(evs, 'x'));
  assert.equal(s.solved, null);
  assert.equal(s.unscored, true);
  assert.equal(s.verified, 1);
  assert.equal(s.tokS, 5);
  const withSummary = V.parseJsonl('{"ev":"run_end","t":3,"cfg":"x","summary":{"tasks":2,"solved_test_inputs":null}}').events;
  assert.equal(V.configSummary(V.buildRun(withSummary, 'x')).unscored, true);
});

test('taskOutcome letters', () => {
  const [a, b] = runsAB;
  const out = (r, id) => V.taskOutcome(r.run.tasks.find((t) => t.id === id));
  assert.equal(out(a, '135a2760'), 'S');
  assert.equal(out(a, '136b0064'), 'F');
  assert.equal(out(a, '142ca369'), '-');
  assert.equal(out(a, '16b78196'), '-');
  assert.equal(out(b, '135a2760'), 'V');
  assert.equal(out(b, '142ca369'), 'S');
  assert.equal(V.taskOutcome(null), '-');
});

test('outcomeMatrix has one row per task and one column per config', () => {
  const m = V.outcomeMatrix(runsAB);
  assert.deepEqual(m.cfgs, ['default', 'fresh']);
  assert.equal(m.rows.length, 5);
  assert.deepEqual(m.rows.map((r) => [r.id, r.outcomes.join('')]), [
    ['135a2760', 'SV'], ['136b0064', 'FF'], ['142ca369', '-S'], ['16b78196', '--'], ['1818057f', '--']]);
});

test('parseBench reads the launcher rows, arrays and wrapped objects', () => {
  const row = { name: 'agent_a', mode: 'agent', tasks: 12, wall_s: 600, tokens: 90000, tok_s: 150.5, verified: 7, score: [4, 12] };
  const rows = V.parseBench(JSON.stringify([row, { name: 'direct', tasks: 12, wall_s: 90, tokens: 5, tok_s: 0, verified: null, score: null }]));
  assert.equal(rows.length, 2);
  assert.equal(rows[0].solved, 4);
  assert.equal(rows[0].solvedOf, 12);
  assert.equal(rows[0].tokS, 150.5);
  assert.equal(rows[0].source, 'bench');
  assert.equal(rows[1].verified, null);
  assert.equal(rows[1].solved, null);
  assert.equal(rows[1].unscored, true);
  assert.equal(V.parseBench(JSON.stringify({ results: [row] })).length, 1);
  const plain = V.parseBench(JSON.stringify([{ name: 'p', tasks: 3, solved: 2, verified: 2, tokens: 1, tok_s: 1, wall_s: 1 }]));
  assert.equal(plain[0].solved, 2);
  assert.equal(V.normalizeBenchRow({ name: 'q', solved: '3/5' }).solvedOf, 5);
  assert.equal(V.normalizeBenchRow({ name: 'e', error: 'boom' }).error, 'boom');
});

test('parseBench rejects traces and junk', () => {
  assert.equal(V.parseBench(sample), null);
  assert.equal(V.parseBench('{"ev":"run_end","t":1,"results":[{"name":"x"}]}'), null);
  assert.equal(V.parseBench('[]'), null);
  assert.equal(V.parseBench('[{"nope":1}]'), null);
  assert.equal(V.parseBench('not json'), null);
});

test('comparisonRows lists traces first and bench rows only without a trace', () => {
  const bench = V.parseBench(JSON.stringify([
    { name: 'fresh', tasks: 99, tokens: 1 }, { name: 'direct_only', tasks: 12, tokens: 7, score: [1, 12] }]));
  const rows = V.comparisonRows(runsAB, bench);
  assert.deepEqual(rows.map((r) => r.name), ['default', 'fresh', 'direct_only']);
  assert.equal(rows[1].tasks, 5);
  assert.equal(rows[1].hasTrace, true);
  assert.equal(rows[2].hasTrace, false);
  assert.equal(rows[2].solved, 1);
  assert.equal(V.comparisonRows([], bench).length, 2);
});

test('tileInfo picks the final attempt, else the first train input', () => {
  const done = V.tileInfo(runsAB[0].run.tasks.find((t) => t.id === '135a2760'));
  assert.equal(done.gridKind, 'attempt');
  assert.equal(done.status, 'verified');
  assert.equal(done.turns, 4);
  assert.equal(done.tokens, 6400 + 900 + 5200 + 160);
  const open = V.tileInfo(runsAB[0].run.tasks.find((t) => t.id === '142ca369'));
  assert.equal(open.gridKind, 'input');
  assert.equal(open.status, 'running');
  const queued = V.tileInfo(runsAB[0].run.tasks.find((t) => t.id === '1818057f'));
  assert.equal(queued.grid, null);
  assert.equal(queued.status, 'queued');
  assert.equal(queued.turns, 0);
});
