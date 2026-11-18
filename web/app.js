/* Sifarish front-end. Vanilla JS, no build step, no dependencies. */

const API = "";

const state = {
  token: localStorage.getItem("sifarish_token") || null,
  user: null,
  view: "feed",
  feedMode: "personal",
  items: [],
  ratings: new Map(), // item_id -> value
  catalogueLoaded: false,
};

/* ------------------------------------------------------------------ utils */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

let toastTimer;
function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 2600);
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body) headers["Content-Type"] = "application/json";
  if (state.token) headers["Authorization"] = `Bearer ${state.token}`;

  const res = await fetch(API + path, { ...options, headers });

  if (res.status === 401 && state.token) {
    signOut(false);
    throw new Error("Your session expired. Please sign in again.");
  }
  if (res.status === 204) return null;

  const text = await res.text();
  let data = null;
  if (text) { try { data = JSON.parse(text); } catch { data = { detail: text }; } }

  if (!res.ok) {
    let detail = data?.detail;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join(", ");
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}

/* ------------------------------------------------------------------ views */

function showView(name) {
  state.view = name;
  $$(".view").forEach((v) => { v.hidden = v.id !== `view-${name}`; });
  $$(".navlink").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === name);
  });
  if (name === "catalogue" && !state.catalogueLoaded) loadCatalogue();
  if (name === "admin") loadAdmin();
  // Always refetch the feed when it comes back into view. Ratings are written
  // from the catalogue, so the feed is stale by definition once you have been
  // anywhere else -- an earlier version showed "no ratings yet" to a user who
  // had just rated three titles.
  if (name === "feed") loadFeed();
  window.scrollTo({ top: 0, behavior: "instant" });
}

/* -------------------------------------------------------------- auth UI */

function renderAuthState() {
  const btn = $("#auth-btn");
  if (state.user) {
    // "Admin" was shown twice: once as the nav link and once as the user's
    // display name, because the seeded admin account is called "Admin".
    // Anyone signed in gets a "Sign out" control, with the name as a tooltip.
    btn.textContent = "Sign out";
    btn.title = `${state.user.display_name} · ${state.user.email}`;
    btn.onclick = () => signOut(true);
  } else {
    btn.textContent = "Sign in";
    btn.onclick = () => openAuth("login");
  }
  $("#nav-admin").hidden = !(state.user && state.user.is_admin);
}

function openAuth(mode) {
  const isLogin = mode === "login";
  $("#modal-body").innerHTML = `
    <h2>${isLogin ? "Welcome back" : "Create an account"}</h2>
    <p class="sub">${isLogin
      ? "Sign in to see recommendations built from your own ratings."
      : "Rate a few titles and the feed adapts as you go."}</p>
    <form id="auth-form">
      ${isLogin ? "" : `
        <label class="field">Display name
          <input type="text" name="display_name" required maxlength="120" placeholder="Your name" />
        </label>`}
      <label class="field">Email
        <input type="email" name="email" required placeholder="you@example.com" />
      </label>
      <label class="field">Password
        <input type="password" name="password" required minlength="8"
               placeholder="${isLogin ? "Your password" : "At least 8 characters, letters and numbers"}" />
      </label>
      <p class="err" id="auth-err"></p>
      <button class="btn btn-primary" type="submit">${isLogin ? "Sign in" : "Create account"}</button>
      <div class="row">
        <button type="button" class="link" id="switch-auth">
          ${isLogin ? "Need an account? Register" : "Already registered? Sign in"}
        </button>
      </div>
    </form>
    <div class="demo-creds">
      Demo account: <code>demo@sifarish.pk</code> / <code>demo12345</code><br />
      Admin: <code>admin@sifarish.pk</code> / <code>admin12345</code>
    </div>`;
  $("#modal").hidden = false;
  $("#switch-auth").onclick = () => openAuth(isLogin ? "register" : "login");
  $("#auth-form").onsubmit = (e) => submitAuth(e, isLogin);
}

async function submitAuth(event, isLogin) {
  event.preventDefault();
  const form = event.target;
  const err = $("#auth-err");
  const button = form.querySelector("button[type=submit]");
  err.textContent = "";
  button.disabled = true;

  const body = {
    email: form.email.value.trim(),
    password: form.password.value,
  };
  if (!isLogin) body.display_name = form.display_name.value.trim();

  try {
    // isLogin was inverted here: a "sign in" went to /register and a
    // "register" went to /login. Registration sends display_name, so the
    // login attempt failed validation with "Field required".
    const data = await api(
      isLogin ? "/api/auth/login" : "/api/auth/register",
      { method: "POST", body: JSON.stringify(body) }
    );
    // The register endpoint returns a token too, so both paths land here.
    state.token = data.access_token;
    state.user = data.user;
    localStorage.setItem("sifarish_token", state.token);
    $("#modal").hidden = true;
    renderAuthState();
    await loadRatings();
    await loadFeed();
    toast(isLogin ? "Signed in" : "Account created");
  } catch (error) {
    // Registering with an email that already exists: sign them in instead of
    // leaving them stuck on a form telling them they already have an account.
    if (!isLogin && /already exists/i.test(error.message)) {
      err.textContent = "That email is registered — signing you in instead…";
      try {
        const data = await api("/api/auth/login", {
          method: "POST", body: JSON.stringify(body),
        });
        state.token = data.access_token;
        state.user = data.user;
        localStorage.setItem("sifarish_token", state.token);
        $("#modal").hidden = true;
        renderAuthState();
        await loadRatings();
        await loadFeed();
        toast("Signed in");
        return;
      } catch (retryError) {
        err.textContent = retryError.message;
        return;
      }
    }
    err.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function signOut(announce) {
  state.token = null;
  state.user = null;
  state.ratings.clear();
  localStorage.removeItem("sifarish_token");
  renderAuthState();
  loadFeed();
  if (announce) toast("Signed out");
}

/* ------------------------------------------------------------------ feed */

const BLEND_LABELS = {
  item_cf: "Item-item CF",
  user_cf: "User-user CF",
  content: "Content (TF-IDF)",
};

function renderBlend(blend, historySize) {
  const bars = $("#blend-bars");
  const order = ["item_cf", "user_cf", "content"];
  bars.innerHTML = order.map((key) => {
    const pct = Math.round((blend[key] || 0) * 100);
    return `
      <div class="blend-row">
        <div class="blend-label">
          <span>${BLEND_LABELS[key]}</span><span>${pct}%</span>
        </div>
        <div class="blend-track">
          <div class="blend-fill ${key.replace("_cf", "").replace("_", "")}"
               style="width:${pct}%"></div>
        </div>
      </div>`;
  }).join("");

  const hist = $("#blend-history");
  hist.textContent = historySize === 0
    ? "no ratings yet"
    : `${historySize} rating${historySize === 1 ? "" : "s"} on record`;

  const note = $("#blend-note");
  if (!state.user) {
    note.textContent =
      "Signed out, so this is pure content + popularity. Sign in to personalise it.";
  } else if (historySize === 0) {
    note.textContent =
      "You have no ratings yet, so every signal is content-based. Rate a few titles from the catalogue.";
  } else if (blend.content > 0.6) {
    note.textContent =
      "Your ratings are still few, so content carries most of the weight. Rate more and collaborative filtering takes over.";
  } else if (blend.item_cf >= 0.5) {
    note.textContent =
      "Enough history now that item-item collaborative filtering leads — people who liked what you liked.";
  } else {
    note.textContent =
      "A mix of collaborative and content signals, weighted by how much history you have.";
  }
}

function itemCard(item, opts = {}) {
  const { score, reason, source, showRate = true, similarity } = opts;
  const myRating = state.ratings.get(item.id) || 0;

  const sourcePill = source
    ? `<span class="pill source-${esc(source)}">${esc(source.replace("_", " "))}</span>`
    : "";

  const scoreRow = score !== undefined
    ? `<div class="score-row">
         <span class="muted" style="margin:0">score</span>
         <div class="score-track"><div class="score-fill" style="width:${Math.min(score * 100, 100).toFixed(0)}%"></div></div>
         <span style="font-variant-numeric:tabular-nums">${score.toFixed(3)}</span>
       </div>`
    : "";

  const simRow = similarity !== undefined
    ? `<div class="score-row">
         <span class="muted" style="margin:0">similarity</span>
         <div class="score-track"><div class="score-fill" style="width:${Math.min(similarity * 100, 100).toFixed(0)}%"></div></div>
         <span style="font-variant-numeric:tabular-nums">${similarity.toFixed(3)}</span>
       </div>`
    : "";

  const avg = item.average_rating !== null && item.average_rating !== undefined
    ? `★ ${item.average_rating.toFixed(1)} · ${item.rating_count} rating${item.rating_count === 1 ? "" : "s"}`
    : "no ratings yet";

  return `
    <article class="card">
      <div class="card-top">
        <h3>${esc(item.title)}</h3>
        ${sourcePill}
      </div>
      <p class="card-desc">${esc(item.description || "No description.")}</p>
      ${reason ? `<p class="reason">${esc(reason)}</p>` : ""}
      ${scoreRow}${simRow}
      <div class="card-meta">
        <span>${esc(item.category)}${item.year ? ` · ${item.year}` : ""}</span>
        <span>${avg}</span>
      </div>
      ${showRate ? `
        <div class="card-meta">
          <div class="stars" data-item="${item.id}">
            ${[1, 2, 3, 4, 5].map((n) =>
              `<button class="star ${n <= myRating ? "on" : ""}" data-value="${n}"
                       aria-label="Rate ${n} out of 5">★</button>`).join("")}
          </div>
          ${myRating ? `<span>you rated ${myRating}</span>` : `<span>not rated</span>`}
        </div>` : ""}
    </article>`;
}

async function loadFeed() {
  const grid = $("#feed-grid");
  const empty = $("#feed-empty");
  grid.innerHTML = '<div class="skeleton-card"></div><div class="skeleton-card"></div><div class="skeleton-card"></div>';
  empty.hidden = true;

  try {
    let data;
    if (state.feedMode === "cold" || !state.user) {
      data = await api("/api/recommendations/cold-start?k=12");
    } else {
      data = await api("/api/recommendations/me?k=12");
    }

    renderBlend(data.blend, data.history_size);

    if (!data.recommendations.length) {
      grid.innerHTML = "";
      empty.hidden = false;
      empty.textContent = state.user
        ? "You have rated everything in the catalogue — nothing left to recommend."
        : "Sign in to see recommendations.";
      return;
    }

    grid.innerHTML = data.recommendations
      .map((r) => itemCard(r.item, { score: r.score, reason: r.reason, source: r.source }))
      .join("");

    $("#feed-title").textContent = state.feedMode === "cold"
      ? "What a brand-new visitor sees"
      : (state.user ? "Your recommendations" : "Cold start (signed out)");
  } catch (error) {
    grid.innerHTML = "";
    empty.hidden = false;
    empty.textContent = error.message;
  }
}

/* ------------------------------------------------------------- catalogue */

async function loadCatalogue() {
  const grid = $("#catalogue-grid");
  grid.innerHTML = '<div class="skeleton-card"></div><div class="skeleton-card"></div><div class="skeleton-card"></div>';
  try {
    if (!state.items.length) {
      state.items = await api("/api/items?limit=200");
    }
    const cats = [...new Set(state.items.map((i) => i.category))].sort();
    const select = $("#category-filter");
    if (select.options.length <= 1) {
      select.innerHTML = '<option value="">All genres</option>' +
        cats.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
    }
    state.catalogueLoaded = true;
    renderCatalogue();
  } catch (error) {
    grid.innerHTML = "";
    $("#catalogue-empty").hidden = false;
    $("#catalogue-empty").textContent = error.message;
  }
}

function renderCatalogue() {
  const q = $("#search").value.trim().toLowerCase();
  const cat = $("#category-filter").value;
  const grid = $("#catalogue-grid");
  const empty = $("#catalogue-empty");

  let items = state.items;
  if (cat) items = items.filter((i) => i.category === cat);
  if (q) {
    items = items.filter((i) =>
      `${i.title} ${i.category} ${i.description} ${i.tags}`.toLowerCase().includes(q)
    );
  }

  if (!items.length) {
    grid.innerHTML = "";
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  grid.innerHTML = items.map((i) => itemCard(i)).join("");
}

/* ----------------------------------------------------------------- admin */

async function loadAdmin() {
  try {
    const [stats, evaluation] = await Promise.all([
      api("/api/admin/stats"),
      api("/api/admin/evaluate?k=10"),
    ]);

    $("#admin-kpis").innerHTML = [
      ["Users", stats.users],
      ["Catalogue", stats.items],
      ["Ratings", stats.ratings],
      ["Coverage", `${(stats.catalogue_coverage * 100).toFixed(1)}%`],
    ].map(([k, v]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");

    const rows = [
      ["Precision@10", "precision_at_k", "precision_pct"],
      ["Recall@10", "recall_at_k", "recall_pct"],
      ["NDCG@10", "ndcg_at_k", "ndcg_pct"],
      ["Catalogue coverage", "coverage", "coverage_pct"],
    ];
    $("#eval-table tbody").innerHTML = rows.map(([label, key, liftKey]) => {
      const lift = evaluation.lift[liftKey];
      const cls = lift >= 0 ? "lift-pos" : "lift-neg";
      const sign = lift >= 0 ? "+" : "";
      return `<tr>
        <td>${label}</td>
        <td>${evaluation.model[key].toFixed(4)}</td>
        <td>${evaluation.baseline[key].toFixed(4)}</td>
        <td class="${cls}">${sign}${lift.toFixed(1)}%</td>
      </tr>`;
    }).join("");

    $("#eval-meta").textContent =
      `${evaluation.split} · k = ${evaluation.k} · ` +
      `${evaluation.model.n_users_evaluated} users evaluated · ` +
      `popularity bias ${(evaluation.model.popularity_bias * 100).toFixed(1)}% of slots go to the top 10% of items`;

    const maxCat = Math.max(...stats.top_categories.map((c) => c.count), 1);
    $("#admin-cats").innerHTML = stats.top_categories.map((c) => `
      <li>
        <div class="bl"><span>${esc(c.category)}</span><span>${c.count}</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:${(c.count / maxCat * 100).toFixed(0)}%"></div></div>
      </li>`).join("");

    $("#admin-items").innerHTML = stats.most_rated.map((i) => `
      <li><span>${esc(i.title)}</span><span class="r">${i.rating_count} · ★ ${i.average}</span></li>
    `).join("");
  } catch (error) {
    toast(error.message);
  }
}

/* ---------------------------------------------------------------- ratings */

async function loadRatings() {
  state.ratings.clear();
  if (!state.user) return;
  try {
    const rows = await api("/api/ratings");
    rows.forEach((r) => state.ratings.set(r.item_id, r.value));
  } catch { /* a signed-out or expired session simply has no ratings */ }
}

async function rate(itemId, value) {
  if (!state.user) {
    openAuth("login");
    return;
  }
  try {
    await api("/api/ratings", {
      method: "PUT",
      body: JSON.stringify({ item_id: itemId, value }),
    });
    state.ratings.set(itemId, value);
    toast(`Rated ${value} — feed updated`);

    // Refresh whichever grid the click came from. The feed is reloaded by
    // showView() when the user navigates back to it.
    if (state.view === "catalogue") {
      state.items = [];
      state.catalogueLoaded = false;
      await loadCatalogue();
    } else {
      await loadFeed();
    }
  } catch (error) {
    toast(error.message);
  }
}

/* ------------------------------------------------------------------ wire */

document.addEventListener("click", (event) => {
  const nav = event.target.closest(".navlink");
  if (nav) { showView(nav.dataset.view); return; }

  const seg = event.target.closest(".seg-btn");
  if (seg) {
    $$(".seg-btn").forEach((b) => b.classList.remove("active"));
    seg.classList.add("active");
    state.feedMode = seg.dataset.mode;
    loadFeed();
    return;
  }

  const star = event.target.closest(".star");
  if (star) {
    const itemId = Number(star.closest(".stars").dataset.item);
    rate(itemId, Number(star.dataset.value));
    return;
  }

  if (event.target.id === "modal-close" || event.target.id === "modal") {
    $("#modal").hidden = true;
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") $("#modal").hidden = true;
});

document.addEventListener("DOMContentLoaded", async () => {
  $("#search").addEventListener("input", renderCatalogue);
  $("#category-filter").addEventListener("change", renderCatalogue);

  if (state.token) {
    try {
      state.user = await api("/api/auth/me");
    } catch { state.token = null; localStorage.removeItem("sifarish_token"); }
  }
  renderAuthState();
  await loadRatings();
  // showView("feed") already triggers loadFeed(); calling it here too would
  // fire two identical requests on every page load.
  showView("feed");
});
