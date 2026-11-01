// Browser smoke test for the static viewer: loads the sample traces in headless Chromium and fails on any
// page error, external request, or missing panel. Needs Playwright resolvable from NODE_PATH (it is not a
// project dependency); tests/test_agent_watch_browser.py skips when it is absent.
'use strict';
const path = require('path');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const sample = (n) => path.join(root, 'sample', n);

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  const external = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('request', (r) => { const u = r.url(); if (!u.startsWith('file://') && !u.startsWith('data:')) external.push(u); });
  await page.goto('file://' + path.join(root, 'index.html'));
  await page.setInputFiles('#file', [sample('sample_trace.jsonl'), sample('sample_trace_b.jsonl')]);
  await page.waitForTimeout(400);
  const watch = await page.evaluate(() => ({
    status: document.querySelector('#status').textContent,
    left: document.querySelector('#left').children.length,
    center: document.querySelector('#center').children.length,
    right: document.querySelector('#right').children.length,
  }));
  await page.click('#btn-prev'); await page.click('#btn-next');
  await page.click('#tab-wall'); await page.waitForTimeout(300);
  const wall = await page.evaluate(() => ({
    panels: document.querySelectorAll('#wall > *').length,
    tiles: document.querySelectorAll('#wall .tile').length,
    rows: document.querySelectorAll('#wall table tbody tr').length,
  }));
  await page.setViewportSize({ width: 390, height: 844 });
  const phone = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth);
  console.log(JSON.stringify({ watch, wall, phoneNoHScroll: phone, errors, external }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
