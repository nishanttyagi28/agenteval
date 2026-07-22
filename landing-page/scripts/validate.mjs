import { access, readFile } from "node:fs/promises";
import { constants } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import vm from "node:vm";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "dist");
const html = await readFile(join(dist, "index.html"), "utf8");
const css = await readFile(join(dist, "styles.css"), "utf8");
const script = await readFile(join(dist, "script.js"), "utf8");
const failures = [];

const check = (condition, message) => {
  if (!condition) failures.push(message);
};

const values = (pattern) => [...html.matchAll(pattern)].map((match) => match[1]);
const ids = values(/\sid=["']([^"']+)["']/g);
const hrefs = values(/\shref=["']([^"']+)["']/g);
const sources = values(/\ssrc=["']([^"']+)["']/g);

check(/<html\s[^>]*lang=["']en["']/i.test(html), "html must declare lang=en");
check(/<meta\s[^>]*name=["']viewport["']/i.test(html), "viewport metadata is required");
check(/<meta\s[^>]*name=["']description["']/i.test(html), "description metadata is required");
check((html.match(/<h1(?:\s|>)/gi) || []).length === 1, "page must contain exactly one h1");
check(/<main\s/i.test(html), "page must contain a main landmark");
check(/<nav\s[^>]*aria-label=/i.test(html), "navigation landmarks need labels");
check(/class=["'][^"']*skip-link/i.test(html), "page needs a skip link");
check(!html.includes("http://"), "all remote resources must use HTTPS");
check(new Set(ids).size === ids.length, "element ids must be unique");

for (const href of hrefs) {
  if (href.startsWith("#")) {
    check(ids.includes(href.slice(1)), `internal link does not resolve: ${href}`);
  } else if (/^https:\/\//.test(href)) {
    try {
      new URL(href);
    } catch {
      failures.push(`invalid external URL: ${href}`);
    }
  } else if (!href.startsWith("mailto:")) {
    const path = href.split(/[?#]/)[0].replace(/^\.\//, "");
    try {
      await access(join(dist, path), constants.R_OK);
    } catch {
      failures.push(`local link does not resolve: ${href}`);
    }
  }
}

for (const source of sources.filter((value) => !/^https:\/\//.test(value))) {
  const path = source.split(/[?#]/)[0].replace(/^\.\//, "");
  try {
    await access(join(dist, path), constants.R_OK);
  } catch {
    failures.push(`local asset does not resolve: ${source}`);
  }
}

for (const image of html.match(/<img\b[^>]*>/gi) || []) {
  check(/\salt=["'][^"']+["']/i.test(image), `image needs descriptive alt text: ${image}`);
  check(/\swidth=["']\d+["']/i.test(image), "images need an explicit width");
  check(/\sheight=["']\d+["']/i.test(image), "images need an explicit height");
}

for (const section of html.match(/<section\b[^>]*>/gi) || []) {
  const labelledBy = section.match(/aria-labelledby=["']([^"']+)["']/i)?.[1];
  check(Boolean(labelledBy), `section needs aria-labelledby: ${section}`);
  if (labelledBy) check(ids.includes(labelledBy), `section label does not resolve: ${labelledBy}`);
}

for (const button of html.match(/<button\b[^>]*>/gi) || []) {
  check(/\stype=["']button["']/i.test(button), `button needs type=button: ${button}`);
  check(/\saria-label=["'][^"']+["']/i.test(button), `button needs an aria-label: ${button}`);
}

check(css.includes("@media (max-width: 1024px)"), "tablet breakpoint is required");
check(css.includes("@media (max-width: 760px)"), "mobile breakpoint is required");
check(css.includes("prefers-reduced-motion: reduce"), "reduced-motion support is required");
check(css.includes(":focus-visible"), "visible keyboard focus styles are required");

const syntax = spawnSync(process.execPath, ["--check", join(dist, "script.js")], {
  encoding: "utf8",
});
check(syntax.status === 0, `script syntax check failed: ${syntax.stderr}`);

const listeners = [];
const anchor = { addEventListener: (type, handler) => listeners.push([type, handler]) };
const classList = { add() {}, remove() {}, toggle() {} };
const navigation = { classList, querySelectorAll: () => [anchor] };
const toggleAttributes = new Map([["aria-expanded", "false"]]);
const navigationToggle = {
  addEventListener: (type, handler) => listeners.push([type, handler]),
  getAttribute: (name) => toggleAttributes.get(name),
  setAttribute: (name, value) => toggleAttributes.set(name, value),
};
const status = { textContent: "" };
const year = { textContent: "" };
const copyLabel = { textContent: "Copy" };
const copyButton = {
  dataset: { copy: "pip install nishanttyagi-agenteval" },
  addEventListener: (type, handler) => listeners.push([type, handler]),
  querySelector: () => copyLabel,
};
const errors = [];
const context = {
  console: { ...console, error: (...args) => errors.push(args.join(" ")) },
  Date,
  document: {
    querySelector: (selector) => ({
      "#primary-navigation": navigation,
      ".nav-toggle": navigationToggle,
      "#copy-status": status,
      "#current-year": year,
    })[selector] || null,
    querySelectorAll: () => [copyButton],
    getElementById: () => null,
  },
  navigator: { clipboard: { writeText: async () => undefined } },
  window: { setTimeout: (handler) => handler() },
};

try {
  vm.runInNewContext(script, context, { filename: "script.js" });
  const clickHandlers = listeners.filter(([type]) => type === "click").map(([, handler]) => handler);
  for (const handler of clickHandlers) await handler();
} catch (error) {
  failures.push(`render smoke test threw: ${error.stack || error}`);
}
check(errors.length === 0, `render smoke test logged console errors: ${errors.join("; ")}`);
check(/^\d{4}$/.test(year.textContent), "footer year script did not render");

if (failures.length) {
  console.error(`Validation failed with ${failures.length} issue(s):`);
  failures.forEach((failure) => console.error(`- ${failure}`));
  process.exit(1);
}

console.log(`Validated ${hrefs.length} links, ${ids.length} anchors, accessibility structure, responsive CSS, and console-clean rendering`);
