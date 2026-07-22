import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { createStaticServer } from "./server-lib.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const output = join(root, "test-results", "visual");
const server = createStaticServer(join(root, "dist"));
const viewports = {
  desktop: { width: 1440, height: 900 },
  tablet: { width: 834, height: 1112 },
  mobile: { width: 390, height: 844 },
};

await mkdir(output, { recursive: true });
await new Promise((resolve, reject) => {
  server.once("error", reject);
  server.listen(4173, "127.0.0.1", resolve);
});

const browser = await chromium.launch();
try {
  for (const [name, viewport] of Object.entries(viewports)) {
    const page = await browser.newPage({ viewport });
    await page.route("https://img.shields.io/**", (route) =>
      route.fulfill({
        status: 200,
        contentType: "image/svg+xml",
        body: '<svg xmlns="http://www.w3.org/2000/svg" width="108" height="20"></svg>',
      }),
    );
    await page.goto("http://127.0.0.1:4173", { waitUntil: "domcontentloaded" });
    await page.screenshot({ path: join(output, `${name}.png`), fullPage: true });
    await page.close();
  }
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}

console.log(`Captured ${Object.keys(viewports).length} responsive screenshots in test-results/visual`);
