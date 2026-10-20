#!/usr/bin/env node
/*
 * Livive showcase recorder.
 *
 * Produces a consistent demo bundle for a project repo:
 *
 *   demos/latest/demo.webm        ~25-30s silent 16:10 screen recording
 *   demos/latest/thumb.webp       1280x800 WebP card thumbnail
 *   demos/latest/poster.webp      960x600 WebP poster
 *   demos/latest/manifest.json    labels, timings, git revision
 *   demos/latest/gallery/*.png    labeled stills captured during the demo
 *
 * The project-specific parts live in scripts/showcase.config.mjs. The runner
 * starts a local static server (or uses --url), drives a headless Chromium
 * through the configured beats, and encodes the result with ffmpeg.
 *
 * Usage:
 *   node scripts/record-showcase.mjs
 *   node scripts/record-showcase.mjs --url https://trot.livive.net/
 *   node scripts/record-showcase.mjs --duration 30000 --keep-raw
 */

import http from "node:http";
import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".htm": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".webmanifest": "application/manifest+json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".u8": "application/octet-stream",
  ".bin": "application/octet-stream",
  ".webm": "video/webm",
  ".mp3": "audio/mpeg",
  ".wav": "audio/wav",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".ttf": "font/ttf",
  ".map": "application/json; charset=utf-8",
};

const args = parseArgs(process.argv.slice(2));
const configPath = path.resolve(ROOT, args.config || "scripts/showcase.config.mjs");
const config = (await import(pathToFileURL(configPath).href)).default;
const VIEW = config.viewport || { width: 1280, height: 800 };
const OUT = path.resolve(ROOT, args.out || config.out || "demos/latest");
const GALLERY = path.join(OUT, "gallery");
const DURATION_MS = Number(args.duration || config.durationMs || 30000);
const KEEP_RAW = Boolean(args.keepRaw);
const URL_OVERRIDE = args.url || config.url || process.env.SHOWCASE_URL || null;

const log = (...m) => console.log("[showcase]", ...m);

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next && !next.startsWith("--")) {
      out[key] = next;
      i++;
    } else {
      out[key] = true;
    }
  }
  out.keepRaw = out["keep-raw"] || out.keepRaw;
  return out;
}

function startServer(dir, spa) {
  const root = path.resolve(ROOT, dir);
  const server = http.createServer((req, res) => {
    let urlPath;
    try {
      urlPath = decodeURIComponent((req.url || "/").split("?")[0]);
    } catch {
      urlPath = "/";
    }
    let filePath;
    if (urlPath === "/") {
      filePath = path.join(root, "index.html");
    } else {
      filePath = path.join(root, urlPath);
      if (urlPath.endsWith("/")) filePath = path.join(filePath, "index.html");
    }
    if (!filePath.startsWith(root)) {
      res.writeHead(403);
      res.end("Forbidden");
      return;
    }
    fs.stat(filePath, (err, stat) => {
      if (err || !stat.isFile()) {
        if (spa) filePath = path.join(root, "index.html");
        else {
          res.writeHead(404);
          res.end("Not found");
          return;
        }
      }
      if (filePath.endsWith(path.sep)) filePath = path.join(filePath, "index.html");
      const ext = path.extname(filePath).toLowerCase();
      res.writeHead(200, {
        "Content-Type": MIME[ext] || "application/octet-stream",
        "Cache-Control": "no-cache",
        "Access-Control-Allow-Origin": "*",
      });
      fs.createReadStream(filePath).on("error", () => {
        res.writeHead(500);
        res.end("Read error");
      }).pipe(res);
    });
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      resolve({ server, url: `http://127.0.0.1:${server.address().port}/` });
    });
  });
}

async function loadChromium() {
  if (process.env.SHOWCASE_PLAYWRIGHT) {
    const mod = await import(pathToFileURL(process.env.SHOWCASE_PLAYWRIGHT).href);
    return mod.chromium;
  }
  try {
    return (await import("playwright")).chromium;
  } catch {
    try {
      return (await import("@playwright/test")).chromium;
    } catch {
      throw new Error("playwright is not installed; run npm ci");
    }
  }
}

async function launchBrowser(chromium) {
  const attempts = [];
  if (process.env.SHOWCASE_CHROME) attempts.push({ headless: true, executablePath: process.env.SHOWCASE_CHROME });
  attempts.push({ headless: true });
  attempts.push({ headless: true, channel: "chrome" });
  let last;
  for (const opts of attempts) {
    try {
      return await chromium.launch(opts);
    } catch (err) {
      last = err;
    }
  }
  throw last;
}

async function injectCursor(page) {
  await page.evaluate(() => {
    if (document.getElementById("showcase-cursor")) return;
    const c = document.createElement("div");
    c.id = "showcase-cursor";
    Object.assign(c.style, {
      position: "fixed",
      top: "-40px",
      left: "-40px",
      width: "16px",
      height: "16px",
      borderRadius: "50%",
      background: "rgba(56, 189, 248, 0.78)",
      border: "2px solid #fff",
      boxShadow: "0 0 12px rgba(56, 189, 248, 0.9)",
      pointerEvents: "none",
      zIndex: "2147483647",
      transform: "translate(-50%, -50%)",
      transition: "width .12s ease, height .12s ease, background-color .12s ease",
    });
    document.documentElement.appendChild(c);
    window.__showcaseCursor = (x, y) => {
      c.style.left = `${x}px`;
      c.style.top = `${y}px`;
    };
  });
}

async function move(page, x, y, steps = 18) {
  await page.mouse.move(x, y, { steps }).catch(() => {});
  await page.evaluate(({ x, y }) => window.__showcaseCursor?.(x, y), { x, y }).catch(() => {});
  await page.waitForTimeout(40);
}

async function drag(page, box, dx, dy) {
  const x = box ? box.x + box.width / 2 : VIEW.width / 2;
  const y = box ? box.y + box.height / 2 : VIEW.height / 2;
  await move(page, x, y);
  await page.mouse.down();
  for (let i = 1; i <= 12; i++) {
    await page.mouse.move(x + (dx * i) / 12, y + (dy * i) / 12);
    await page.waitForTimeout(18);
  }
  await page.mouse.up();
}

async function clickLocator(page, locator) {
  await locator.scrollIntoViewIfNeeded({ timeout: 3000 }).catch(() => {});
  const box = await locator.boundingBox().catch(() => null);
  if (box) await move(page, box.x + box.width / 2, box.y + box.height / 2);
  await locator.click({ timeout: 5000 });
  await page.waitForTimeout(120);
}

async function doAction(page, action) {
  if (!action) return;
  if (Array.isArray(action)) {
    for (const a of action) await doAction(page, a);
    return;
  }
  try {
    if (action.wait) return await page.waitForTimeout(action.wait);
    if (action.click) return await clickLocator(page, page.locator(action.click).first());
    if (action.clickText) {
      const escaped = String(action.clickText).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const byRole = page.getByRole("button", { name: new RegExp(escaped, "i") }).first();
      if (await byRole.count()) return await clickLocator(page, byRole);
      return await clickLocator(page, page.getByText(new RegExp(escaped, "i")).first());
    }
    if (action.fill) return await page.locator(action.fill.sel).first().fill(action.fill.text, { timeout: 5000 });
    if (action.select) return await page.locator(action.select.sel).first().selectOption(action.select.value);
    if (action.press) return await page.keyboard.press(action.press);
    if (action.hover) return await page.locator(action.hover).first().hover({ timeout: 4000 });
    if (action.scrollTo !== undefined) return await page.evaluate((y) => window.scrollTo({ top: y, behavior: "instant" }), action.scrollTo);
    if (action.scroll !== undefined) return await page.evaluate((y) => window.scrollBy(0, y), action.scroll);
    if (action.scrollToSelector) return await page.locator(action.scrollToSelector).first().scrollIntoViewIfNeeded({ timeout: 4000 });
    if (action.mouse) return await move(page, action.mouse.x, action.mouse.y);
    if (action.waitFor) return await page.locator(action.waitFor).first().waitFor({ state: "visible", timeout: 15000 });
    if (action.waitForText) return await page.getByText(action.waitForText, { exact: false }).first().waitFor({ state: "visible", timeout: 20000 });
    if (action.waitForFunction) return await page.waitForFunction(action.waitForFunction, null, { timeout: action.timeout || 30000 });
    if (action.eval) {
      if (typeof action.eval === "function") return await page.evaluate(action.eval);
      return await page.evaluate(action.eval);
    }
    if (action.drag) {
      const box = action.drag.sel ? await page.locator(action.drag.sel).first().boundingBox().catch(() => null) : null;
      return await drag(page, box, action.drag.dx, action.drag.dy);
    }
  } catch (err) {
    console.warn(`[showcase] skipped action ${JSON.stringify(action)}: ${String(err.message || err).split("\n")[0]}`);
  }
}

function slug(s) {
  return String(s || "shot")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48) || "shot";
}

async function toWebp(page, pngPath, width, height, quality, outPath) {
  const b64 = (await fsp.readFile(pngPath)).toString("base64");
  const dataUrl = await page.evaluate(
    async ({ b64, width, height, quality }) => {
      const img = new Image();
      img.src = `data:image/png;base64,${b64}`;
      await img.decode();
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      // cover-crop
      const scale = Math.max(width / img.width, height / img.height);
      const dw = img.width * scale;
      const dh = img.height * scale;
      ctx.drawImage(img, (width - dw) / 2, (height - dh) / 2, dw, dh);
      return canvas.toDataURL("image/webp", quality);
    },
    { b64, width, height, quality },
  );
  await fsp.writeFile(outPath, Buffer.from(dataUrl.split(",")[1], "base64"));
}

function run(cmd, argv) {
  return new Promise((resolve) => {
    const p = spawn(cmd, argv, { stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    p.stderr.on("data", (d) => (stderr += d));
    p.on("error", () => resolve(false));
    p.on("close", (code) => {
      if (code !== 0) console.warn(`[showcase] ${cmd} exited ${code}: ${stderr.slice(0, 300)}`);
      resolve(code === 0);
    });
  });
}

async function main() {
  await fsp.rm(OUT, { recursive: true, force: true });
  await fsp.mkdir(GALLERY, { recursive: true });

  const tmp = await fsp.mkdtemp(path.join(os.tmpdir(), "livive-showcase-"));
  const chromium = await loadChromium();
  const browser = await launchBrowser(chromium);
  let server = null;
  let baseUrl = URL_OVERRIDE;
  if (!baseUrl) {
    const serve = config.serve || { dir: "." };
    server = await startServer(serve.dir, Boolean(serve.spa));
    baseUrl = server.url;
  }
  log(`source ${baseUrl}`);

  const context = await browser.newContext({
    viewport: VIEW,
    deviceScaleFactor: 1,
    ignoreHTTPSErrors: true,
    recordVideo: { dir: tmp, size: VIEW },
  });
  const page = await context.newPage();
  const video = page.video();
  const consoleErrors = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => consoleErrors.push(String(e)));

  let gallery = [];
  let thumbShot = null;
  const started = Date.now();
  let tourStarted = started;

  try {
    await page.goto(baseUrl, { waitUntil: "load", timeout: 60000 });
    await page.waitForLoadState("networkidle", { timeout: 10000 }).catch(() => {});
    await injectCursor(page);
    await page.waitForTimeout(config.settleMs ?? 1200);

    if (config.ready) await config.ready(page);
    if (config.prelude) await config.prelude(page);

    tourStarted = Date.now();
    let shotNo = 0;
    for (const beat of config.beats || []) {
      log(`beat ${beat.label || beat.shot || shotNo + 1}`);
      await doAction(page, beat.action);
      if (beat.wait) await page.waitForTimeout(beat.wait);
      if (beat.shot) {
        shotNo += 1;
        const name = `${String(shotNo).padStart(2, "0")}-${slug(beat.shot)}.png`;
        const file = path.join(GALLERY, name);
        await page.screenshot({ path: file, type: "png" });
        const item = { label: beat.label || beat.shot, src: `gallery/${name}`, file };
        gallery.push(item);
        if (beat.thumb || config.thumbFrom === name || config.thumbFrom === beat.shot) thumbShot = item;
      }
    }

    const budgetStart = config.trimIntro ? tourStarted : started;
    const remaining = DURATION_MS - (Date.now() - budgetStart);
    if (remaining > 0) await page.waitForTimeout(remaining);

    // Thumbnail and poster are encoded from the chosen gallery still.
    // A project can opt into an element crop so chrome (status bars, nav) is
    // excluded from the card thumbnail.
    let thumbSource = null;
    if (config.thumbSelector) {
      const el = page.locator(config.thumbSelector).first();
      if (await el.count()) {
        thumbSource = path.join(OUT, ".thumb-source.png");
        await el.screenshot({ path: thumbSource, type: "png" }).catch(() => {});
      }
    }
    if (!thumbSource) {
      if (!thumbShot) thumbShot = gallery[0];
      thumbSource = thumbShot ? thumbShot.file : null;
    }
    if (thumbSource && fs.existsSync(thumbSource)) {
      await toWebp(page, thumbSource, 1280, 800, 0.82, path.join(OUT, "thumb.webp"));
      await toWebp(page, thumbSource, 960, 600, 0.8, path.join(OUT, "poster.webp"));
      if (thumbSource.endsWith(".thumb-source.png")) await fsp.rm(thumbSource, { force: true });
    }

    const rawDurationMs = Date.now() - started;
    const introMs = config.trimIntro ? Math.max(0, tourStarted - started - 500) : 0;
    await context.close();
    const raw = video ? await video.path() : null;

    const demoPath = path.join(OUT, "demo.webm");
    let encoded = false;
    if (raw && fs.existsSync(raw)) {
      const ffmpeg = process.env.SHOWCASE_FFMPEG || "ffmpeg";
      const seek = introMs > 0 ? ["-ss", (introMs / 1000).toFixed(2)] : [];
      encoded = await run(ffmpeg, [
        "-y", "-i", raw, ...seek, "-an",
        "-c:v", "libvpx", "-b:v", "1200k", "-crf", "34",
        "-pix_fmt", "yuv420p",
        "-vf", `scale=${VIEW.width}:${VIEW.height}`,
        demoPath,
      ]);
      if (!encoded) await fsp.copyFile(raw, demoPath);
    }
    const durationMs = Math.max(1000, rawDurationMs - introMs);

    const manifest = {
      key: config.key,
      source: baseUrl,
      captured_at: new Date().toISOString(),
      git: {
        sha: process.env.GITHUB_SHA || null,
        ref: process.env.GITHUB_REF_NAME || null,
      },
      viewport: VIEW,
      duration_ms: durationMs,
      video: "demo.webm",
      thumb: "thumb.webp",
      poster: "poster.webp",
      video_bytes: fs.existsSync(demoPath) ? fs.statSync(demoPath).size : 0,
      encoded_with_ffmpeg: encoded,
      gallery: gallery.map(({ label, src }) => ({ label, src })),
      console_errors: consoleErrors.slice(0, 10),
    };
    await fsp.writeFile(path.join(OUT, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");

    const size = fs.existsSync(demoPath) ? (fs.statSync(demoPath).size / 1024 / 1024).toFixed(2) : "0";
    log(`wrote ${path.relative(ROOT, OUT)} (${gallery.length} stills, demo ${size} MB, ${(durationMs / 1000).toFixed(1)}s)`);
    if (consoleErrors.length) log(`console errors: ${consoleErrors.length} (see manifest)`);
  } finally {
    await context.close().catch(() => {});
    await browser.close().catch(() => {});
    if (server) server.server.close();
    if (!KEEP_RAW) await fsp.rm(tmp, { recursive: true, force: true });
    else log(`raw video kept in ${tmp}`);
  }
}

main().catch((err) => {
  console.error("[showcase] failed:", err);
  process.exit(1);
});
