(() => {
  "use strict";

  const navigation = document.querySelector("#primary-navigation");
  const navigationToggle = document.querySelector(".nav-toggle");
  const copyStatus = document.querySelector("#copy-status");
  const currentYear = document.querySelector("#current-year");

  const closeNavigation = () => {
    if (!navigation || !navigationToggle) return;
    navigation.classList.remove("open");
    navigationToggle.setAttribute("aria-expanded", "false");
    navigationToggle.setAttribute("aria-label", "Open navigation");
  };

  if (navigation && navigationToggle) {
    navigationToggle.addEventListener("click", () => {
      const willOpen = navigationToggle.getAttribute("aria-expanded") !== "true";
      navigation.classList.toggle("open", willOpen);
      navigationToggle.setAttribute("aria-expanded", String(willOpen));
      navigationToggle.setAttribute("aria-label", willOpen ? "Close navigation" : "Open navigation");
    });

    navigation.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", closeNavigation);
    });
  }

  if (currentYear) currentYear.textContent = String(new Date().getFullYear());

  document.querySelectorAll("[data-copy], [data-copy-target]").forEach((button) => {
    button.addEventListener("click", async () => {
      const targetId = button.dataset.copyTarget;
      const target = targetId ? document.getElementById(targetId) : null;
      const value = button.dataset.copy || target?.innerText || target?.textContent || "";
      const label = button.querySelector("span");

      try {
        await navigator.clipboard.writeText(value.trim());
        if (label) label.textContent = "Copied";
        if (copyStatus) copyStatus.textContent = "Copied to clipboard";
        window.setTimeout(() => {
          if (label) label.textContent = "Copy";
        }, 1600);
      } catch {
        if (copyStatus) copyStatus.textContent = "Copy failed. Select and copy the text manually.";
      }
    });
  });

  // Cosmetic demo only: these static example values do not execute AgentEval or call an API.
  const terminalDemo = document.querySelector("[data-terminal-demo]");
  const replayButton = terminalDemo?.querySelector("[data-run-demo]");

  if (terminalDemo && replayButton) {
    const demoLines = [...terminalDemo.querySelectorAll("[data-demo-text]")];
    const metricTracks = [...terminalDemo.querySelectorAll("[data-metric]")];
    const demoTiming = terminalDemo.querySelector("[data-demo-timing]");
    const demoGate = terminalDemo.querySelector("[data-demo-gate]");
    const demoResult = terminalDemo.querySelector("[data-demo-result]");
    const demoStatus = terminalDemo.querySelector("[data-demo-status]");
    const buttonLabel = replayButton.querySelector("[data-demo-button-label]");
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let activeRun = 0;

    const pause = (duration, runId) => new Promise((resolve) => {
      window.setTimeout(() => resolve(runId === activeRun), duration);
    });

    const setMetric = (track, value) => {
      const label = track.parentElement?.querySelector("[data-metric-value]");
      const displayValue = value === 100 ? "100%" : `${value.toFixed(1)}%`;
      track.style.setProperty("--progress", `${value}%`);
      track.setAttribute("aria-valuenow", String(value));
      if (label) label.textContent = displayValue;
    };

    const resetDemo = () => {
      demoLines.forEach((line) => {
        line.textContent = "";
        line.classList.remove("is-typing");
      });
      metricTracks.forEach((track) => setMetric(track, 0));
      demoTiming?.classList.remove("is-visible");
      demoGate?.classList.remove("is-visible");
      demoResult?.classList.remove("is-visible");
      if (demoStatus) demoStatus.textContent = "";
      terminalDemo.dataset.demoState = "running";
      terminalDemo.setAttribute("aria-busy", "true");
      if (buttonLabel) buttonLabel.textContent = "Restart demo";
    };

    const finishDemo = () => {
      demoGate?.classList.add("is-visible");
      demoResult?.classList.add("is-visible");
      terminalDemo.dataset.demoState = "complete";
      terminalDemo.setAttribute("aria-busy", "false");
      if (buttonLabel) buttonLabel.textContent = "Replay demo";
      if (demoStatus) demoStatus.textContent = "Static AgentEval demonstration complete. Regression gate: pass.";
    };

    const showFinalState = () => {
      demoLines.forEach((line) => {
        line.textContent = line.dataset.demoText || "";
      });
      demoTiming?.classList.add("is-visible");
      metricTracks.forEach((track) => setMetric(track, Number(track.dataset.metric)));
      finishDemo();
    };

    const typeLine = async (line, runId) => {
      const value = line.dataset.demoText || "";
      line.classList.add("is-typing");
      for (const character of value) {
        if (runId !== activeRun) return false;
        line.textContent += character;
        if (!(await pause(24, runId))) return false;
      }
      line.classList.remove("is-typing");
      return pause(110, runId);
    };

    const animateMetrics = (runId) => new Promise((resolve) => {
      const duration = 720;
      const startedAt = performance.now();
      const frame = (now) => {
        if (runId !== activeRun) {
          resolve(false);
          return;
        }
        const progress = Math.min((now - startedAt) / duration, 1);
        const eased = 1 - ((1 - progress) ** 3);
        metricTracks.forEach((track) => setMetric(track, Number(track.dataset.metric) * eased));
        if (progress < 1) {
          window.requestAnimationFrame(frame);
          return;
        }
        resolve(true);
      };
      window.requestAnimationFrame(frame);
    });

    const runDemo = async () => {
      const runId = ++activeRun;
      resetDemo();

      if (reduceMotion.matches) {
        showFinalState();
        return;
      }

      for (const line of demoLines) {
        if (!(await typeLine(line, runId))) return;
        if (line.dataset.demoText === "Evaluation complete") {
          demoTiming?.classList.add("is-visible");
        }
      }

      if (!(await animateMetrics(runId))) return;
      if (!(await pause(180, runId))) return;
      finishDemo();
    };

    replayButton.addEventListener("click", runDemo);
    runDemo();
  }
})();
