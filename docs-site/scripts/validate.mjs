import { access, readFile } from "node:fs/promises";
import { constants } from "node:fs";
import { join, dirname } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "dist");
const pages = ["index.html", "cli-reference.html", "adapter-guide.html", "comparison.html"];

const css = await readFile(join(dist, "styles.css"), "utf8");
const script = await readFile(join(dist, "script.js"), "utf8");
const failures = [];

const check = (condition, message) => {
  if (!condition) failures.push(message);
};

const values = (html, pattern) => [...html.matchAll(pattern)].map((match) => match[1]);

for (const page of pages) {
  const html = await readFile(join(dist, page), "utf8");
  const ids = values(html, /\sid=["']([^"']+)["']/g);
  const hrefs = values(html, /\shref=["']([^"']+)["']/g);
  const sources = values(html, /\ssrc=["']([^"']+)["']/g);

  check(/<html\s[^>]*lang=["']en["']/i.test(html), `${page}: html must declare lang=en`);
  check(/<meta\s[^>]*name=["']viewport["']/i.test(html), `${page}: viewport metadata is required`);
  check(
    /<meta\s[^>]*name=["']description["']/i.test(html),
    `${page}: description metadata is required`
  );
  check((html.match(/<h1(?:\s|>)/gi) || []).length === 1, `${page}: page must contain exactly one h1`);
  check(/<main\s/i.test(html), `${page}: page must contain a main landmark`);
  check(/<nav\s[^>]*aria-label=/i.test(html), `${page}: navigation landmarks need labels`);
  check(/class=["'][^"']*skip-link/i.test(html), `${page}: page needs a skip link`);
  check(!html.includes("http://"), `${page}: all remote resources must use HTTPS`);
  check(new Set(ids).size === ids.length, `${page}: element ids must be unique`);

  for (const href of hrefs) {
    if (href.startsWith("#")) {
      check(ids.includes(href.slice(1)), `${page}: internal link does not resolve: ${href}`);
    } else if (/^https:\/\//.test(href)) {
      try {
        new URL(href);
      } catch {
        failures.push(`${page}: invalid external URL: ${href}`);
      }
    } else if (!href.startsWith("mailto:")) {
      const [path, anchor] = href.split("#");
      const cleanPath = path.replace(/^\.\//, "");
      if (cleanPath) {
        try {
          await access(join(dist, cleanPath), constants.R_OK);
        } catch {
          failures.push(`${page}: local link does not resolve: ${href}`);
        }
      }
      if (anchor && cleanPath && cleanPath !== page) {
        // Cross-page anchor: verify the target page actually declares that id.
        try {
          const targetHtml = await readFile(join(dist, cleanPath), "utf8");
          const targetIds = values(targetHtml, /\sid=["']([^"']+)["']/g);
          check(targetIds.includes(anchor), `${page}: cross-page anchor does not resolve: ${href}`);
        } catch {
          failures.push(`${page}: cross-page link target unreadable: ${href}`);
        }
      }
    }
  }

  for (const source of sources.filter((value) => !/^https:\/\//.test(value))) {
    const path = source.split(/[?#]/)[0].replace(/^\.\//, "");
    try {
      await access(join(dist, path), constants.R_OK);
    } catch {
      failures.push(`${page}: local asset does not resolve: ${source}`);
    }
  }

  for (const image of html.match(/<img\b[^>]*>/gi) || []) {
    check(/\salt=["'][^"']*["']/i.test(image), `${page}: image needs alt text: ${image}`);
    check(/\swidth=["']\d+["']/i.test(image), `${page}: images need an explicit width`);
    check(/\sheight=["']\d+["']/i.test(image), `${page}: images need an explicit height`);
  }
}

check(css.includes("@media (max-width: 1024px)"), "tablet breakpoint is required");
check(css.includes("@media (max-width: 760px)"), "mobile breakpoint is required");
check(css.includes("prefers-reduced-motion: reduce"), "reduced-motion support is required");
check(css.includes(":focus-visible"), "visible keyboard focus styles are required");

const syntax = spawnSync(process.execPath, ["--check", join(dist, "script.js")], {
  encoding: "utf8",
});
check(syntax.status === 0, `script syntax check failed: ${syntax.stderr}`);

if (failures.length) {
  console.error(`Validation failed with ${failures.length} issue(s):`);
  failures.forEach((failure) => console.error(`- ${failure}`));
  process.exit(1);
}

console.log(`Validated ${pages.length} pages: links, anchors, accessibility structure, and responsive CSS.`);
