import { copyFile, mkdir, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const output = join(root, "dist");
const pages = ["index.html", "cli-reference.html", "adapter-guide.html", "comparison.html"];
const assets = [...pages, "styles.css", "script.js"];

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await Promise.all(assets.map((asset) => copyFile(join(root, asset), join(output, asset))));
await writeFile(join(output, ".nojekyll"), "", "utf8");

console.log(`Built ${assets.length} static assets (${pages.length} pages) in docs-site/dist`);
