# AgentEval landing page

A framework-light static site built with semantic HTML, CSS, and minimal
JavaScript. The production artifact has no runtime package dependencies.

## Local development

```bash
npm install
npx playwright install chromium
npm test
npm run serve
```

Open `http://127.0.0.1:4173`. `npm test` builds the site, validates local links
and accessibility structure, executes the page script in a lightweight smoke
environment, then runs Chromium and axe checks at desktop, tablet, and mobile
viewports.

Use `npm run capture` to write full-page responsive screenshots under the
ignored `test-results/visual/` directory.

## Deployment

The `Deploy AgentEval landing page` workflow builds and tests the page before
uploading `landing-page/dist` as a GitHub Pages artifact. In repository
**Settings → Pages**, set **Source** to **GitHub Actions**. A push to `main` that
changes this directory or the workflow will deploy the site; the workflow can
also be started manually from the Actions tab.
