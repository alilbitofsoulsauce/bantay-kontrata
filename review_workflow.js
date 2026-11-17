const reviewStatuses = new Map([
  ["in_review","In review"],
  ["needs_documents","Needs documents"],
  ["reviewed","Reviewed"],
  ["follow_up","Follow up"]
]);
const reviewLimits = {reviewer:80,evidence_url:650,notes:2000};
const reviewSchema = "bantay-kontrata-tagum-reviews-v1";
const reviewStorageKey = reviewSchema + ":" + data.snapshot_id;
const reviewDate = new Intl.DateTimeFormat("en-PH",{
  dateStyle:"medium",timeStyle:"short",timeZone:"Asia/Manila"
});
const reviewKey = (kind,d) => kind + ":" + d[reviewViews[kind].key];
const reviewRecords = new Map(Object.keys(reviewViews).flatMap(kind =>
  data[kind].records.map(d => [reviewKey(kind,d),{kind,d}])
));
let reviewStore = null;
let reviewLog = new Map();

function evidenceLink(value) {
  if (!value) return "";
  try {
    const url = new URL(value);
    return url.protocol === "https:" && !url.username && !url.password ? url.href : null;
  } catch {
    return null;
  }
}

function serializeReviews(entries) {
  return JSON.stringify({
    schema:reviewSchema,snapshot_id:data.snapshot_id,entries:[...entries.values()]
  },null,2) + "\n";
}

function parseReviews(text) {
  const backup = JSON.parse(text);
  if (backup?.schema !== reviewSchema || !Array.isArray(backup.entries)) {
    throw new Error("Choose a Bantay-Kontrata JSON backup.");
  }
  if (backup.snapshot_id !== data.snapshot_id) {
    throw new Error("This backup belongs to a different data snapshot. Open its matching dashboard.");
  }
  const entries = new Map();
  for (const item of backup.entries) {
    if (!item || !reviewRecords.has(item.key) || entries.has(item.key) ||
        !reviewStatuses.has(item.status) || !Number.isFinite(Date.parse(item.updated_at)) ||
        new Date(item.updated_at).toISOString() !== item.updated_at ||
        Object.entries(reviewLimits).some(([field,limit]) => typeof item[field] !== "string" || item[field].length > limit) ||
        evidenceLink(item.evidence_url) === null) {
      throw new Error("The backup contains an invalid or duplicate review. No reviews were imported.");
    }
    const {key,status,reviewer,evidence_url,notes,updated_at} = item;
    entries.set(key,{key,status,reviewer,evidence_url,notes,updated_at});
  }
  return entries;
}

function loadReviews() {
  if (reviewStore) {
    try {
      const raw = reviewStore.getItem(reviewStorageKey);
      reviewLog = raw === null ? new Map() : parseReviews(raw);
    } catch {
      reviewStore = null;
    }
  }
  return reviewLog;
}

function persistReviews(entries) {
  reviewLog = entries;
  if (reviewStore) {
    try {
      reviewStore.setItem(reviewStorageKey,serializeReviews(entries));
    } catch {
      reviewStore = null;
    }
  }
  paintQueueStatuses("award");
  paintQueueStatuses("tender");
  renderSavedReviews();
}

function paintQueueStatuses(kind) {
  for (const badge of id(kind + "Table").querySelectorAll("[data-review-key]")) {
    const review = reviewLog.get(badge.dataset.reviewKey);
    badge.textContent = review ? " · " + reviewStatuses.get(review.status) : "";
  }
}

function reviewInput(form,input,label,name,value,long = false) {
  const wrapper = node("label",label,long ? "full" : "");
  input.name = name;
  input.value = value;
  if (name in reviewLimits) input.maxLength = reviewLimits[name];
  wrapper.append(input);
  form.append(wrapper);
}

function drawCaseReview(container,kind,d) {
  const key = reviewKey(kind,d);
  const initial = loadReviews().get(key);
  const panel = node("details","","review-entry");
  const summary = node("summary",initial ? "Review · " + reviewStatuses.get(initial.status) : "Record review");
  const form = node("form");
  const status = node("select");
  reviewStatuses.forEach((label,value) => option(status,label,value));
  reviewInput(form,status,"Review status","status",initial?.status || "in_review");
  reviewInput(form,node("input"),"Reviewer initials (optional)","reviewer",initial?.reviewer || "");
  const evidence = node("input");
  evidence.type = "url";
  evidence.placeholder = "https://";
  reviewInput(form,evidence,"Evidence link (optional)","evidence_url",initial?.evidence_url || "",true);
  const notes = node("textarea");
  notes.rows = 3;
  notes.placeholder = "What did you verify, or which document is still needed?";
  reviewInput(form,notes,"Notes","notes",initial?.notes || "",true);
  const save = node("button","Save review","minor primary");
  save.type = "submit";
  const actions = node("div","","actions full");
  const feedback = node("p","","small review-feedback full");
  feedback.setAttribute("role","status");
  actions.append(save);
  form.append(actions,feedback);
  const source = node("p","Record only the facts you verified and any follow-up needed.","small");
  panel.append(summary,source,form);
  container.append(panel);

  let original = initial;
  form.addEventListener("submit",event => {
    event.preventDefault();
    const enteredLink = evidence.value.trim();
    if (evidenceLink(enteredLink) === null) {
      feedback.textContent = "Use an HTTPS evidence link, or leave that field blank.";
      evidence.focus();
      return;
    }
    const next = new Map(loadReviews());
    if (JSON.stringify(next.get(key)) !== JSON.stringify(original)) {
      feedback.textContent = "This review changed in another tab. Reopen the row to load the saved version.";
      return;
    }
    original = {
      key,status:status.value,reviewer:form.elements.reviewer.value.trim(),
      evidence_url:enteredLink,notes:notes.value.trim(),updated_at:new Date().toISOString()
    };
    next.set(key,original);
    persistReviews(next);
    summary.textContent = "Review · " + reviewStatuses.get(original.status);
    feedback.textContent = reviewStore ? "Review saved in this browser." :
      "Review saved for this session. Download a backup in Saved reviews before closing.";
  });
}

function openSavedReview(key) {
  const {kind,d} = reviewRecords.get(key);
  const view = reviewViews[kind];
  if (kind === "award") {
    id("awardSearch").value = "";
    id("awardCategory").value = "";
    id("flagsOnly").checked = false;
    id("awardLinked").checked = false;
    view.state.month = "";
    drawTimeline();
  } else {
    id("tenderSearch").value = "";
    id("tenderMode").value = "";
    id("tenderFlagsOnly").checked = false;
  }
  selectTab(id(view.verb + "-tab"));
  view.state.page = Math.floor(data[kind].records.indexOf(d) / reviewPageSize);
  renderReview(kind,d[view.key]);
  const panel = id(view.inspector).querySelector(".review-entry");
  panel.open = true;
  panel.querySelector("summary").focus();
  panel.scrollIntoView({block:"nearest"});
}

function sortedReviews() {
  return [...loadReviews().values()].sort((a,b) => b.updated_at.localeCompare(a.updated_at));
}

function renderSavedReviews() {
  const entries = sortedReviews();
  const selected = id("savedStatus").value;
  const visible = entries.filter(entry => !selected || entry.status === selected);
  const table = id("savedCases");
  table.replaceChildren();
  id("savedCount").textContent = number.format(visible.length) + " of " + number.format(entries.length) + " saved reviews";
  id("reviewStorage").textContent = reviewStore ?
    "Reviews are saved in this browser for this data snapshot. Download a backup to resume on another device." :
    "Browser storage is unavailable. Download a backup before closing this page.";
  id("reviewCsv").disabled = id("reviewBackup").disabled = entries.length === 0;
  for (const entry of visible) {
    const {kind,d} = reviewRecords.get(entry.key);
    const row = node("tr");
    const record = addCell(row,"");
    const button = node("button",reviewViews[kind].title(d),"case-button");
    button.type = "button";
    button.onclick = () => openSavedReview(entry.key);
    record.append(button,node("span",entry.key,"case-sub"));
    addCell(row,reviewStatuses.get(entry.status));
    addCell(row,entry.reviewer || "—");
    addCell(row,reviewDate.format(new Date(entry.updated_at)));
    table.append(row);
  }
  if (!visible.length) {
    const row = node("tr");
    addCell(row,entries.length ? "No saved reviews match this status." :
      "Open a record in either review queue to save your first review.").colSpan = 4;
    table.append(row);
  }
}

function csvCell(value) {
  let text = String(value ?? "");
  if (/^[\s\uFEFF]*[=+\-@\t\r\n]/u.test(text)) text = "'" + text;
  return '"' + text.replaceAll('"','""') + '"';
}

function reviewCsv(entries) {
  const columns = [
    "snapshot_id","case_id","queue","title","record_date","bid_reference","official_award_reference",
    "amount_or_budget_php","isolation_forest_score","above_cutoff","review_status","reviewer",
    "official_notice_url","evidence_url","notes","updated_at_utc"
  ];
  const rows = entries.map(entry => {
    const {kind,d} = reviewRecords.get(entry.key);
    return [
      data.snapshot_id,entry.key,kind,reviewViews[kind].title(d),d.award_date || d.published_date,
      d.bid_reference,d.official_award_reference,d.contract_amount ?? d.approved_budget_php,
      d.anomaly_score,d.review,reviewStatuses.get(entry.status),entry.reviewer,
      d.official_notice_link,entry.evidence_url,entry.notes,entry.updated_at
    ];
  });
  return "\uFEFF" + [columns,...rows].map(row => row.map(csvCell).join(",")).join("\r\n") + "\r\n";
}

function downloadReviewFile(content,extension,type) {
  const url = URL.createObjectURL(new Blob([content],{type}));
  const link = node("a");
  link.href = url;
  link.download = "Tagum_reviews_" + data.snapshot_id.slice(0,12) + "." + extension;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url),1000);
}

function initializeReviewWorkflow() {
  try {
    reviewStore = window.localStorage;
    loadReviews();
    if (reviewStore) {
      reviewStore.setItem(reviewStorageKey + ".probe","1");
      reviewStore.removeItem(reviewStorageKey + ".probe");
    }
  } catch {
    reviewStore = null;
  }
  const snapshot = data.coverage;
  id("snapshotBanner").textContent = "Data snapshot " + data.snapshot_id.slice(0,12) +
    " · Official tenders through " + snapshot.latest_tender_publication_date +
    " · Official awards through " + snapshot.latest_official_award_date;
  reviewStatuses.forEach((label,value) => option(id("savedStatus"),label,value));
  id("savedStatus").onchange = renderSavedReviews;
  id("reviewCsv").onclick = () => downloadReviewFile(reviewCsv(sortedReviews()),"csv","text/csv;charset=utf-8");
  id("reviewBackup").onclick = () => downloadReviewFile(serializeReviews(loadReviews()),"json","application/json");
  const file = id("reviewFile");
  id("reviewRestore").onclick = () => file.click();
  file.onchange = async () => {
    const source = file.files[0];
    file.value = "";
    if (!source) return;
    const feedback = id("reviewFeedback");
    try {
      const incoming = parseReviews(await source.text());
      const entries = new Map(loadReviews());
      const before = entries.size;
      for (const [key,entry] of incoming) {
        if (!entries.has(key)) entries.set(key,entry);
      }
      persistReviews(entries);
      const added = entries.size - before;
      feedback.textContent = number.format(added) + " imported. " + number.format(incoming.size - added) +
        " existing reviews kept." + (reviewStore ? "" : " Changes are available for this session.");
    } catch (error) {
      feedback.textContent = error instanceof SyntaxError ? "Choose a valid JSON backup." : error.message;
    }
  };
  window.addEventListener("storage",event => {
    if (event.key === reviewStorageKey || event.key === null) {
      renderSavedReviews();
      paintQueueStatuses("award");
      paintQueueStatuses("tender");
    }
  });
  renderSavedReviews();
}
