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
  assert.match(V.submitDetail(sub), /Example 2: 1 cells wrong/);
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
