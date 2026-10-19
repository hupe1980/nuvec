// Adds a copy button to every code block in the docs.
(function () {
  document.querySelectorAll(".prose pre").forEach((pre) => {
    const button = document.createElement("button");
    button.className = "copy";
    button.type = "button";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code to clipboard");
    button.addEventListener("click", () => {
      navigator.clipboard.writeText(pre.innerText.replace(/Copy$/, "").trimEnd()).then(() => {
        button.textContent = "Copied";
        setTimeout(() => { button.textContent = "Copy"; }, 1500);
      });
    });
    pre.appendChild(button);
  });
})();
