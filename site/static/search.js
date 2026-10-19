// Docs search: loads Zola's elasticlunr index on first use and lists the top matches.
// Press "/" to focus the field, Escape to clear it.
(function () {
  const input = document.getElementById("search");
  const list = document.getElementById("search-results");
  if (!input || !list) return;

  const base = document.querySelector('link[rel="icon"]').href.replace(/favicon\.svg.*$/, "");
  let index = null;
  let loading = null;

  function load() {
    loading = loading || fetch(base + "search_index.en.json")
      .then((r) => r.json())
      .then((data) => { index = elasticlunr.Index.load(data); });
    return loading;
  }

  function escape(text) {
    const div = document.createElement("div");
    div.textContent = text || "";
    return div.innerHTML;
  }

  function render(query) {
    if (!query.trim()) { list.hidden = true; list.innerHTML = ""; return; }
    const hits = index.search(query, {
      bool: "AND",
      expand: true,
      fields: { title: { boost: 3 }, description: { boost: 2 }, body: { boost: 1 } },
    }).slice(0, 8);
    list.innerHTML = hits.length
      ? hits.map((h) => {
          const doc = index.documentStore.getDoc(h.ref);
          return `<li><a href="${h.ref}">${escape(doc.title)}<small>${escape(doc.description || doc.body.slice(0, 90))}</small></a></li>`;
        }).join("")
      : '<li class="empty">No results</li>';
    list.hidden = false;
  }

  input.addEventListener("focus", load, { once: true });
  input.addEventListener("input", () => load().then(() => render(input.value)));
  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { input.value = ""; render(""); input.blur(); }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== input) { e.preventDefault(); input.focus(); }
  });
})();
