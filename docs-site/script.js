(() => {
  "use strict";

  const navigation = document.querySelector("#primary-navigation");
  const navigationToggle = document.querySelector(".nav-toggle");
  const currentYear = document.querySelector("#current-year");

  if (navigation && navigationToggle) {
    navigationToggle.addEventListener("click", () => {
      const willOpen = navigationToggle.getAttribute("aria-expanded") !== "true";
      navigation.classList.toggle("open", willOpen);
      navigationToggle.setAttribute("aria-expanded", String(willOpen));
      navigationToggle.setAttribute("aria-label", willOpen ? "Close navigation" : "Open navigation");
    });
  }

  if (currentYear) {
    currentYear.textContent = String(new Date().getFullYear());
  }
})();
