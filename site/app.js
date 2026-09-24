const labels = {
  ready: "Ready",
  review: "Review",
  used_elsewhere: "Used elsewhere",
  used_by_launch: "Used by launch",
};

const state = { rows: [], tier: "all", query: "" };
const tbody = document.querySelector("#candidate-rows");

function cell(row, text, className = "") {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  row.append(td);
  return td;
}

function render() {
  const filtered = state.rows.filter((item) =>
    (state.tier === "all" || item.Tier === state.tier) &&
    item.Package.toLowerCase().includes(state.query)
  );
  const fragment = document.createDocumentFragment();
  for (const item of filtered) {
    const tr = document.createElement("tr");
    cell(tr, item.Rank, "numeric");
    const packageCell = cell(tr, "");
    const packageLink = document.createElement("a");
    packageLink.className = "package-name";
    packageLink.href = `https://github.com/autowarefoundation/autoware_universe/tree/${encodeURIComponent(window.universeCommit || "main")}/${item.Source_Path.replace(/^src\/universe\/autoware_universe\//, "").split("/").map(encodeURIComponent).join("/")}`;
    packageLink.textContent = item.Package;
    packageCell.append(packageLink);
    cell(tr, item.Score, "numeric score");
    const statusCell = cell(tr, "");
    const badge = document.createElement("span");
    badge.className = `status ${item.Tier}`;
    badge.textContent = labels[item.Tier];
    statusCell.append(badge);
    cell(tr, item.Internal_Recursive_Dependent_Count, "numeric");
    cell(tr, item.Internal_Prerequisite_Count, "numeric");
    const evidence = cell(tr, "", "evidence");
    if (item.Launch_Reachable) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "View launch path";
      const path = document.createElement("span");
      path.className = "path";
      path.textContent = item.Launch_Path;
      details.append(summary, path);
      evidence.append(details);
    } else if (item.Tier === "ready") {
      if (item.Internal_Prerequisite_Count) {
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = "View Universe prerequisites";
        const names = document.createElement("span");
        names.className = "path";
        names.textContent = item.Internal_Prerequisites.split(";").join(", ");
        details.append(summary, names);
        evidence.append(details);
      } else {
        evidence.textContent = item.Standalone ? "No known users or Universe prerequisites" : "No known external use";
      }
    } else if (item.Tier === "review") {
      evidence.textContent = `${item.External_Review_Reference_Count} lower-confidence mention(s)`;
    } else {
      evidence.textContent = `${item.External_Manifest_Dependent_Count} manifest user(s), ${item.External_High_Reference_Count} active reference(s)`;
    }
    fragment.append(tr);
  }
  tbody.replaceChildren(fragment);
  document.querySelector("#result-count").textContent = `${filtered.length} of ${state.rows.length} packages shown`;
  document.querySelector("#empty-state").hidden = filtered.length !== 0;
}

function sourceLink(name, commit) {
  if (!/^[0-9a-f]{40}$/.test(commit || "")) return null;
  const link = document.createElement("a");
  link.href = `https://github.com/autowarefoundation/${name}/commit/${commit}`;
  link.textContent = `${name.replace("autoware_", "")} ${commit.slice(0, 7)}`;
  return link;
}

async function load() {
  try {
    const response = await fetch("data.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    state.rows = data.candidates;
    window.universeCommit = data.metadata.commits.universe;
    document.querySelector("#generated-at").textContent = new Date(data.metadata.generated_at).toLocaleString(undefined, { dateStyle: "long", timeStyle: "short", timeZone: "UTC" }) + " UTC";
    document.querySelector("#source-revision").textContent = window.universeCommit ? `Universe ${window.universeCommit.slice(0, 12)}` : "Source revision unavailable";
    for (const tier of ["all", ...Object.keys(labels)]) {
      document.querySelector(`#count-${tier}`).textContent = tier === "all" ? state.rows.length : (data.metadata.tier_counts[tier] || 0);
    }
    const links = document.querySelector("#source-links");
    for (const [name, commit] of Object.entries({ autoware: data.metadata.commits.autoware, autoware_universe: data.metadata.commits.universe, autoware_launch: data.metadata.commits.launch })) {
      const link = sourceLink(name, commit);
      if (link) links.append(link);
    }
    render();
  } catch (error) {
    document.querySelector("#result-count").textContent = "Could not load candidate data. Download the CSV or try again later.";
    document.querySelector("#generated-at").textContent = "Unavailable";
    console.error(error);
  }
}

document.querySelector("#search").addEventListener("input", (event) => {
  state.query = event.target.value.trim().toLowerCase();
  render();
});
for (const button of document.querySelectorAll("[data-tier]")) {
  button.addEventListener("click", () => {
    state.tier = button.dataset.tier;
    for (const other of document.querySelectorAll("[data-tier]")) {
      const active = other === button;
      other.classList.toggle("active", active);
      other.setAttribute("aria-pressed", String(active));
    }
    render();
  });
}
load();
