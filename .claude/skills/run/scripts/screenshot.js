// Headless-browser screenshot helper for verifying the visualizer (see
// ../SKILL.md). Requires `playwright` installed in the CWD you run this
// from - it is NOT a project dependency (see SKILL.md step 2), so run this
// from a scratch npm project, not from viz/frontend/.
//
// Usage:
//   node screenshot.js <url> <outfile> [--forward N]
//
//   <url>       e.g. http://127.0.0.1:8000/
//   <outfile>   path to write the PNG to
//   --forward N click the replay's Panel A "step >" button N times before
//               capturing (0 = capture the initial dashboard state as-is).
//               The visualizer auto-selects the most recently created run
//               and its first episode on load, so with exactly one run in
//               runs/, no manual selection is needed first.
const { chromium } = require("playwright");

function parseArgs(argv) {
  const [url, outfile, ...rest] = argv;
  let forward = 0;
  for (let i = 0; i < rest.length; i++) {
    if (rest[i] === "--forward") forward = parseInt(rest[++i], 10);
  }
  return { url, outfile, forward };
}

(async () => {
  const { url, outfile, forward } = parseArgs(process.argv.slice(2));
  if (!url || !outfile) {
    console.error("usage: node screenshot.js <url> <outfile> [--forward N]");
    process.exit(1);
  }

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  await page.goto(url, { waitUntil: "networkidle" });

  const forwardBtn = page.locator(".btn-forward").first();
  for (let i = 0; i < forward; i++) {
    await forwardBtn.click();
    await page.waitForTimeout(200); // let the canvas redraw before the next click/screenshot
  }

  await page.screenshot({ path: outfile, fullPage: true });
  await browser.close();
  console.log(`wrote ${outfile}`);
})();
