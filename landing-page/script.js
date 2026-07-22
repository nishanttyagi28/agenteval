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
})();
