/*
 * Precompiles the JSX in backend/index.html into backend/index.prod.html.
 *
 * The browser otherwise has to load and parse 3.14MB of babel.min.js and
 * transform the JSX on every single launch. The compiled page skips both.
 *
 * Uses the copy of Babel already vendored in backend/vendor, so this needs no
 * npm install and no new dependency.
 *
 *   node tools/build_ui.cjs
 *
 * backend/index.html stays the file you edit. main.py only serves the compiled
 * page while it is newer than the source, so a forgotten rebuild falls back to
 * the source rather than serving something stale.
 */

const fs = require("fs");
const path = require("path");

const ROOT    = path.join(__dirname, "..");
const SOURCE  = path.join(ROOT, "backend", "index.html");
const OUTPUT  = path.join(ROOT, "backend", "index.prod.html");
const BABEL   = path.join(ROOT, "backend", "vendor", "babel.min.js");

require(BABEL);
const Babel = globalThis.Babel;
if (!Babel || typeof Babel.transform !== "function") {
  console.error("Could not load Babel from " + BABEL);
  process.exit(1);
}

const html = fs.readFileSync(SOURCE, "utf8");

const scriptRe = /<script type="text\/babel">([\s\S]*?)<\/script>/;
const match = html.match(scriptRe);
if (!match) {
  console.error('No <script type="text/babel"> block found in ' + SOURCE);
  process.exit(1);
}

let compiled;
try {
  compiled = Babel.transform(match[1], { presets: ["react"] }).code;
} catch (err) {
  console.error("JSX failed to compile:\n" + err.message);
  process.exit(1);
}

// A literal </script> inside the compiled JS would close the tag early.
compiled = compiled.replace(/<\/script>/gi, "<\\/script>");

let out = html.replace(scriptRe, '<script>\n' + compiled + '\n</script>');

// The compiled page has no JSX left, so the transformer is dead weight.
out = out.replace(/[ \t]*<script src="\/vendor\/babel\.min\.js"><\/script>\r?\n?/, "");

if (out.includes("babel.min.js")) {
  console.error("babel.min.js is still referenced after the rewrite; aborting.");
  process.exit(1);
}

fs.writeFileSync(OUTPUT, out, "utf8");

const kb = (n) => Math.round(n / 1024) + " KB";
console.log("source   " + path.relative(ROOT, SOURCE) + "  " + kb(html.length));
console.log("compiled " + path.relative(ROOT, OUTPUT) + "  " + kb(out.length));
console.log("jsx " + kb(match[1].length) + " -> js " + kb(compiled.length) +
            ", browser no longer loads 3.14 MB of babel.min.js");
