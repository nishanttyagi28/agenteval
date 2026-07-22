import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const expectedSections = ["top", "workflow", "features", "usage", "stats"];

test.beforeEach(async ({ page }) => {
  await page.route("https://img.shields.io/**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "image/svg+xml",
      body: '<svg xmlns="http://www.w3.org/2000/svg" width="108" height="20"></svg>',
    }),
  );
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
});

test("renders without console errors or horizontal overflow", async ({ page }) => {
  const consoleErrors = [];
  const pageErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.reload({ waitUntil: "domcontentloaded" });
  await expect(page).toHaveTitle("AgentEval — CI for AI agents");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(/Ship agents with/);
  await expect(page.locator("main")).toBeVisible();

  const dimensions = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    content: document.documentElement.scrollWidth,
  }));
  expect(dimensions.content).toBeLessThanOrEqual(dimensions.viewport);
  expect(consoleErrors).toEqual([]);
  expect(pageErrors).toEqual([]);
});

test("terminal demo animates to completion and can be replayed", async ({ page }) => {
  const consoleErrors = [];
  const pageErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.reload({ waitUntil: "domcontentloaded" });

  const demo = page.locator("[data-terminal-demo]");
  const replay = page.getByRole("button", { name: "Run terminal demo" });
  const initialBox = await demo.boundingBox();
  await expect(demo).toHaveAttribute("data-demo-state", "complete", { timeout: 10_000 });
  const completedBox = await demo.boundingBox();
  expect(initialBox).not.toBeNull();
  expect(completedBox).not.toBeNull();
  expect(Math.abs(completedBox.height - initialBox.height)).toBeLessThanOrEqual(1);
  expect(Math.abs(completedBox.width - initialBox.width)).toBeLessThanOrEqual(1);
  await expect(demo.locator('[data-demo-text="agenteval run --agent research_crew"]')).toHaveText(
    "agenteval run --agent research_crew",
  );
  await expect(demo.locator("[data-demo-timing]")).toHaveText("8.42s");
  await expect(demo.locator('[aria-label="Correctness"]')).toHaveAttribute("aria-valuenow", "95.2");
  await expect(demo.locator('[aria-label="Tool accuracy"]')).toHaveAttribute("aria-valuenow", "100");
  await expect(demo.locator('[aria-label="Trajectory match"]')).toHaveAttribute("aria-valuenow", "92.8");
  await expect(demo.locator("[data-demo-gate]")).toHaveClass(/is-visible/);

  await replay.click();
  await expect(demo).toHaveAttribute("data-demo-state", "running");
  await expect(demo.locator('[aria-label="Correctness"]')).toHaveAttribute("aria-valuenow", "0");
  await expect(demo.locator("[data-demo-gate]")).not.toHaveClass(/is-visible/);
  await expect(demo).toHaveAttribute("data-demo-state", "complete", { timeout: 10_000 });
  await expect(demo.getByText("PASS", { exact: true })).toBeVisible();
  expect(consoleErrors).toEqual([]);
  expect(pageErrors).toEqual([]);
});

test("all internal navigation targets exist and copy actions work", async ({ page }) => {
  const hrefs = await page.locator('a[href^="#"]').evaluateAll((links) =>
    [...new Set(links.map((link) => link.getAttribute("href")))],
  );

  for (const href of hrefs) {
    await expect(page.locator(href)).toHaveCount(1);
  }
  for (const id of expectedSections) {
    await expect(page.locator(`#${id}`)).toHaveCount(1);
  }

  await page.getByRole("button", { name: "Copy installation command" }).first().click();
  await expect(page.locator("#copy-status")).toHaveText("Copied to clipboard");
});

test("has semantic landmarks and no detectable accessibility violations", async ({ page }) => {
  await expect(page.getByRole("main")).toHaveCount(1);
  await expect(page.locator("nav")).toHaveCount(2);
  await expect(page.getByRole("heading", { level: 1 })).toHaveCount(1);
  await expect(page.getByRole("img", { name: /GitHub stars/ })).toHaveCount(1);

  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
});

test("navigation behavior matches the active breakpoint", async ({ page }, testInfo) => {
  const toggle = page.getByRole("button", { name: "Open navigation" });
  const navigation = page.getByRole("navigation", { name: "Primary navigation" });

  if (testInfo.project.name !== "mobile") {
    await expect(toggle).toBeHidden();
    await expect(navigation).toBeVisible();
    return;
  }

  await expect(navigation).toBeHidden();
  await toggle.click();
  await expect(navigation).toBeVisible();
  await expect(page.locator(".nav-toggle")).toHaveAttribute("aria-expanded", "true");
  await navigation.getByRole("link", { name: "Features" }).click();
  await expect(navigation).toBeHidden();
  await expect(page).toHaveURL(/#features$/);
});
