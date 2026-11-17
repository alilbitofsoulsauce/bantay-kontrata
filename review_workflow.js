const reviewStatuses = new Map([
  ["in_review","In review"],
  ["needs_documents","Needs documents"],
  ["reviewed","Reviewed"],
  ["follow_up","Follow up"]
]);
const reviewLimits = {reviewer:80,evidence_url:650,evidence_ref:200,notes:2000};
const reviewSchema = "bantay-kontrata-tagum-reviews-v1";
const reviewStorageKey = reviewSchema + ":" + data.snapshot_id;
const serverReviewMode = document.documentElement.dataset.reviewStore === "server";
const reviewDate = new Intl.DateTimeFormat("en-PH",{
  dateStyle:"medium",timeStyle:"short",timeZone:"Asia/Manila"
});
const reviewKey = (kind,d) => kind + ":" + d[reviewViews[kind].key];
const sameReview = (a,b) => JSON.stringify(a) === JSON.stringify(b);
const reviewRecords = new Map(Object.keys(reviewViews).flatMap(kind =>
  data[kind].records.map(d => [reviewKey(kind,d),{kind,d}])
));
let reviewStore = null;
let reviewLog = new Map();
let reviewRevision = null;
let reviewConnection = "connecting";
const reviewDrafts = new Map();
const reviewSaves = new Set();
const reviewSaveMessages = new Map();

function evidenceLink(value) {
  if (!value) return "";
  try {
    const url = new URL(value);
    return url.protocol === "https:" && !url.username && !url.password ? url.href : null;
  } catch {
    return null;
  }
}

function reviewEnvelope(entries) {
  return {schema:reviewSchema,snapshot_id:data.snapshot_id,entries:[...entries]};
}

function serializeReviews(entries) {
  return JSON.stringify(reviewEnvelope(entries.values()),null,2) + "\n";
}

function parseReviews(backup) {
  if (backup?.schema !== reviewSchema || !Array.isArray(backup.entries)) {
    throw new Error("Choose a Bantay-Kontrata JSON backup.");
  }
  if (backup.snapshot_id !== data.snapshot_id) {
    throw new Error("This backup belongs to a different data snapshot. Open its matching dashboard.");
  }
  const entries = new Map();
  for (const item of backup.entries) {
    const {key,status,reviewer,evidence_url,evidence_ref = "",notes,updated_at} = item || {};
    const fields = {reviewer,evidence_url,evidence_ref,notes};
    if (!reviewRecords.has(key) || entries.has(key) ||
        !reviewStatuses.has(status) || !Number.isFinite(Date.parse(updated_at)) ||
        new Date(updated_at).toISOString() !== updated_at ||
        Object.entries(reviewLimits).some(([field,limit]) => typeof fields[field] !== "string" || fields[field].length > limit) ||
        evidenceLink(evidence_url) === null) {
      throw new Error("The backup contains an invalid or duplicate review. No reviews were imported.");
    }
    entries.set(key,{key,status,...fields,updated_at});
  }
  return entries;
}

function loadReviews() {
  if (reviewStore) {
    try {
      const raw = reviewStore.getItem(reviewStorageKey);
      reviewLog = raw === null ? new Map() : parseReviews(JSON.parse(raw));
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
  updateReviewViews();
}

function updateReviewViews() {
  paintQueueStatuses("award");
  paintQueueStatuses("tender");
  renderSavedReviews();
  if (serverReviewMode) {
    for (const view of Object.values(reviewViews)) {
      id(view.inspector).querySelector(".review-entry form")?.dispatchEvent(new Event("reviewstorechange"));
    }
  }
}

async function requestServerReviews(update) {
  let response, payload;
  try {
    response = update ? await fetch("/api/reviews",{
      method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(update)
    }) : await fetch("/api/reviews?snapshot_id=" + encodeURIComponent(data.snapshot_id),{cache:"no-store"});
    payload = await response.json();
  } catch {
    reviewConnection = "unavailable";
    renderSavedReviews();
    throw new Error("Could not reach the app’s reviews database. Try again.");
  }
  if (!response.ok) {
    if (response.status >= 500 || reviewRevision === null) {
      reviewConnection = "unavailable";
      renderSavedReviews();
    }
    const error = new Error(typeof payload?.error === "string" ? payload.error : "The reviews database rejected the request.");
    error.status = response.status;
    throw error;
  }
  let entries;
  try {
    if (!Number.isSafeInteger(payload?.revision) || payload.revision < 0) throw new Error();
    entries = parseReviews(payload);
  } catch {
    reviewConnection = "unavailable";
    renderSavedReviews();
    throw new Error("The app returned an invalid review state. Reload and try again.");
  }
  if (reviewRevision === null || payload.revision >= reviewRevision) {
    reviewRevision = payload.revision;
    reviewLog = entries;
  }
  reviewConnection = "connected";
  updateReviewViews();
  return {entries:reviewLog,revision:reviewRevision};
}

function refreshReviews() {
  return serverReviewMode ? requestServerReviews() : {entries:loadReviews()};
}

async function persistReviewChanges(state,entries,source) {
  if (serverReviewMode) {
    return requestServerReviews({...reviewEnvelope(entries),revision:state.revision,source});
  }
  const next = new Map(state.entries);
  for (const entry of entries) next.set(entry.key,entry);
  persistReviews(next);
  return {entries:next};
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

function reviewFields(entry) {
  return {
    status:entry?.status ?? "in_review",reviewer:entry?.reviewer ?? "",
    evidence_url:entry?.evidence_url ?? "",evidence_ref:entry?.evidence_ref ?? "",notes:entry?.notes ?? ""
  };
}

function warnAboutReviewDrafts(event) {
  event.preventDefault();
  event.returnValue = true;
}

function setReviewDraft(key,draft) {
  if (draft) {
    reviewDrafts.set(key,draft);
    window.addEventListener("beforeunload",warnAboutReviewDrafts);
  } else {
    reviewDrafts.delete(key);
    if (!reviewDrafts.size) window.removeEventListener("beforeunload",warnAboutReviewDrafts);
  }
}

function drawCaseReview(container,kind,d) {
  const key = reviewKey(kind,d);
  const draft = reviewDrafts.get(key);
  let original = draft ? draft.original : loadReviews().get(key);
  const entered = draft?.fields ?? reviewFields(original);
  const panel = node("details","","review-entry");
  panel.open = Boolean(draft);
  const summary = node("summary");
  const form = node("form");
  const status = node("select");
  reviewStatuses.forEach((label,value) => option(status,label,value));
  reviewInput(form,status,"Review status","status",entered.status);
  const reviewer = node("input");
  reviewInput(form,reviewer,"Reviewer initials","reviewer",entered.reviewer);
  const evidence = node("input");
  evidence.type = "url";
  evidence.placeholder = "https://";
  reviewInput(form,evidence,"Evidence link","evidence_url",entered.evidence_url,true);
  const reference = node("input");
  reviewInput(form,reference,"Internal document reference","evidence_ref",entered.evidence_ref,true);
  const notes = node("textarea");
  notes.rows = 3;
  notes.placeholder = "What did you verify, or which document is still needed?";
  reviewInput(form,notes,"Notes","notes",entered.notes,true);
  const save = node("button","Save review","minor primary");
  save.type = "submit";
  const discard = node("button","Discard draft","minor");
  discard.type = "button";
  const actions = node("div","","actions full");
  const pending = node("p","Unsaved changes are kept on this page until you save or discard them.","small full");
  const feedback = node("p","","small review-feedback full");
  feedback.setAttribute("role","status");
  actions.append(save,discard);
  form.append(actions,pending,feedback);
  const source = node("p","For Reviewed and Follow up, enter your initials, explain your decision in Notes, and add an evidence link or internal document reference.","small");
  panel.append(summary,source,form);
  container.append(panel);

  const fieldNames = Object.keys(entered);
  const readFields = () => Object.fromEntries(fieldNames.map(name => [name,form.elements[name].value]));
  const showDraftState = dirty => {
    const label = original ? "Review · " + reviewStatuses.get(original.status) : "Record review";
    summary.textContent = label + (dirty ? " · Unsaved draft" : "");
    discard.hidden = pending.hidden = !dirty;
  };
  const resetForm = saved => {
    original = saved;
    const fields = reviewFields(saved);
    for (const name of fieldNames) form.elements[name].value = fields[name];
    setReviewDraft(key,null);
    showDraftState(false);
  };
  const setSaveMessage = message => {
    if (serverReviewMode) reviewSaveMessages.set(key,message);
    feedback.textContent = message;
  };
  const syncServerForm = () => {
    const saving = reviewSaves.has(key);
    for (const control of form.elements) control.disabled = saving;
    const saved = loadReviews().get(key);
    if (!saving && !reviewDrafts.has(key) && !sameReview(saved,original)) resetForm(saved);
    feedback.textContent = reviewSaveMessages.get(key) ?? "";
  };
  showDraftState(Boolean(draft));
  if (serverReviewMode) {
    form.addEventListener("reviewstorechange",syncServerForm);
    syncServerForm();
  }
  form.addEventListener("input",() => {
    const fields = readFields();
    const dirty = !sameReview(fields,reviewFields(original));
    setReviewDraft(key,dirty ? {original,fields} : null);
    showDraftState(dirty);
    if (serverReviewMode) setSaveMessage("");
  });
  discard.onclick = () => {
    resetForm(loadReviews().get(key));
    setSaveMessage("Draft discarded.");
    notes.focus();
  };
  form.addEventListener("submit",async event => {
    event.preventDefault();
    if (reviewSaves.has(key)) return;
    const fields = Object.fromEntries(Object.entries(readFields()).map(([name,value]) => [name,value.trim()]));
    if (evidenceLink(fields.evidence_url) === null) {
      setSaveMessage("Use an HTTPS evidence link, or leave that field blank.");
      evidence.focus();
      return;
    }
    if ((fields.status === "reviewed" || fields.status === "follow_up") &&
        (!fields.reviewer || !fields.notes || (!fields.evidence_url && !fields.evidence_ref))) {
      setSaveMessage("Enter your initials, explain your decision in Notes, and add an evidence link or internal document reference.");
      const missing = !fields.reviewer ? reviewer : !fields.notes ? notes : reference;
      missing.focus();
      return;
    }
    if (serverReviewMode) {
      reviewSaves.add(key);
      setSaveMessage("Saving review…");
      syncServerForm();
    }
    const warnIfCaseChanged = saved => {
      if (sameReview(saved,original)) return false;
      if (reviewDrafts.has(key)) {
        setSaveMessage("The saved review changed. Copy any draft text you need, then discard the draft to load the latest saved review.");
      } else {
        resetForm(saved);
        setSaveMessage("The saved review changed. The latest version is shown here.");
      }
      return true;
    };
    try {
      const state = await refreshReviews();
      if (warnIfCaseChanged(state.entries.get(key))) return;
      const saved = {key,...fields,updated_at:new Date().toISOString()};
      const next = await persistReviewChanges(state,[saved],"edit");
      resetForm(next.entries.get(key));
      setSaveMessage(serverReviewMode ? "Review saved in the app database." : reviewStore ?
        "Review saved in this browser." : "Review saved for this session. Download a backup in Saved reviews before closing.");
    } catch (error) {
      if (error.status === 409) {
        await requestServerReviews().catch(() => {});
        if (warnIfCaseChanged(loadReviews().get(key))) return;
      }
      setSaveMessage("Review could not be saved. " + error.message);
    } finally {
      if (serverReviewMode) {
        reviewSaves.delete(key);
        updateReviewViews();
      }
    }
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

function sortedReviews(entries = loadReviews()) {
  return [...entries.values()].sort((a,b) => b.updated_at.localeCompare(a.updated_at));
}

function renderSavedReviews() {
  const entries = sortedReviews();
  const selected = id("savedStatus").value;
  const visible = entries.filter(entry => !selected || entry.status === selected);
  const table = id("savedCases");
  table.replaceChildren();
  id("savedCount").textContent = number.format(visible.length) + " of " + number.format(entries.length) + " saved reviews";
  const connectionMessages = {
    connecting:"Connecting to the app’s reviews database…",
    connected:"Reviews for this snapshot are saved in the app database.",
    unavailable:"The app’s reviews database is unavailable. Previously loaded reviews may be out of date."
  };
  id("reviewStorage").textContent = serverReviewMode ? connectionMessages[reviewConnection] : reviewStore ?
    "Reviews are saved in this browser for this data snapshot. Download a backup to resume on another device." :
    "Browser storage is unavailable. Download a backup before closing this page.";
  id("reviewCsv").disabled = id("reviewBackup").disabled = !serverReviewMode && entries.length === 0;
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
    addCell(row,entries.length ? "No saved reviews match this status." : serverReviewMode && reviewConnection !== "connected" ?
      "Saved reviews are currently unavailable." :
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
    "official_notice_url","evidence_url","notes","updated_at_utc","evidence_ref"
  ];
  const rows = entries.map(entry => {
    const {kind,d} = reviewRecords.get(entry.key);
    return [
      data.snapshot_id,entry.key,kind,reviewViews[kind].title(d),d.award_date || d.published_date,
      d.bid_reference,d.official_award_reference,d.contract_amount ?? d.approved_budget_php,
      d.anomaly_score,d.review,reviewStatuses.get(entry.status),entry.reviewer,
      d.official_notice_link,entry.evidence_url,entry.notes,entry.updated_at,entry.evidence_ref
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

async function exportReviews(format) {
  const feedback = id("reviewFeedback");
  try {
    const {entries} = await refreshReviews();
    if (!entries.size) {
      feedback.textContent = "There are no saved reviews to export.";
      return;
    }
    if (format === "csv") {
      downloadReviewFile(reviewCsv(sortedReviews(entries)),"csv","text/csv;charset=utf-8");
    } else {
      downloadReviewFile(serializeReviews(entries),"json","application/json");
    }
  } catch (error) {
    feedback.textContent = "Could not export reviews. " + error.message;
  }
}

function reviewComparison(label,entry) {
  const section = node("section");
  const fields = node("dl");
  for (const [name,value] of [
    ["Status",reviewStatuses.get(entry.status)],["Reviewer",entry.reviewer],
    ["Updated (UTC)",entry.updated_at],
    ["Evidence link",entry.evidence_url],["Internal document reference",entry.evidence_ref],["Notes",entry.notes]
  ]) {
    fields.append(node("dt",name),node("dd",value || "—"));
  }
  section.append(node("h3",label),fields);
  return section;
}

function showReviewConflicts(conflicts) {
  const container = id("reviewConflicts");
  container.replaceChildren();
  if (!conflicts.length) return;

  const heading = node("h3","Conflicting reviews");
  const instructions = node("p","Compare each case before replacing a saved review. Download a backup first if you need a copy of the saved version. A new backup import replaces this comparison list.","small");
  const feedback = id("reviewFeedback");
  container.append(heading,instructions);

  for (const [saved,incoming] of conflicts) {
    const {kind,d} = reviewRecords.get(incoming.key);
    const panel = node("details","","review-entry");
    const title = reviewViews[kind].title(d) + " · " + incoming.key;
    const comparison = node("div","","review-comparison");
    comparison.append(reviewComparison("Saved here",saved),reviewComparison("From backup",incoming));
    const actions = node("div","","actions");
    const keep = node("button","Keep saved review","minor");
    const replace = node("button","Use backup review","minor");
    keep.type = replace.type = "button";
    actions.append(keep,replace);
    panel.append(node("summary",title),comparison,actions);
    container.append(panel);

    const finish = message => {
      if (!panel.isConnected) return;
      const next = panel.nextElementSibling || panel.previousElementSibling;
      panel.remove();
      if (!container.querySelector("details")) container.replaceChildren();
      const summary = next?.querySelector("summary");
      (summary || id("reviewRestore")).focus();
      feedback.textContent = incoming.key + " · " + message;
    };
    const resolve = async useBackup => {
      keep.disabled = replace.disabled = true;
      try {
        const state = await refreshReviews();
        if (!sameReview(state.entries.get(incoming.key),saved)) {
          finish("The saved review changed. Import the backup again to compare the latest version.");
          return;
        }
        if (!useBackup) {
          finish("Kept the saved review.");
          return;
        }
        await persistReviewChanges(state,[incoming],"backup");
        finish(serverReviewMode ? "Saved the review from the backup in the app database." : reviewStore ?
          "Saved the review from the backup." : "Used the review from the backup for this session.");
      } catch (error) {
        if (panel.isConnected) feedback.textContent = "Could not apply the comparison. " + error.message;
      } finally {
        keep.disabled = replace.disabled = false;
      }
    };
    keep.onclick = () => resolve(false);
    replace.onclick = () => resolve(true);
  }
}

function initializeReviewWorkflow() {
  if (!serverReviewMode) {
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
  }
  const snapshot = data.coverage;
  id("snapshotBanner").textContent = "Retrospective snapshot " + data.snapshot_id.slice(0,12) +
    " · Official tenders through " + snapshot.latest_tender_publication_date +
    " · Official awards through " + snapshot.latest_official_award_date +
    ". Confirm current status and deadlines in the official notice and amendments.";
  reviewStatuses.forEach((label,value) => option(id("savedStatus"),label,value));
  id("savedStatus").onchange = renderSavedReviews;
  id("reviewStorage").setAttribute("role","status");
  id("reviewCsv").onclick = () => exportReviews("csv");
  id("reviewBackup").onclick = () => exportReviews("json");
  const file = id("reviewFile");
  const restore = id("reviewRestore");
  restore.onclick = () => file.click();
  file.onchange = async () => {
    const source = file.files[0];
    file.value = "";
    if (!source) return;
    const feedback = id("reviewFeedback");
    restore.disabled = true;
    try {
      const incoming = parseReviews(JSON.parse(await source.text()));
      const state = await refreshReviews();
      const added = [];
      let identical = 0;
      const conflicts = [];
      for (const [key,entry] of incoming) {
        const saved = state.entries.get(key);
        if (!saved) {
          added.push(entry);
        } else if (sameReview(saved,entry)) {
          identical += 1;
        } else {
          conflicts.push([saved,entry]);
        }
      }
      if (added.length) await persistReviewChanges(state,added,"backup");
      showReviewConflicts(conflicts);
      feedback.textContent = number.format(added.length) + " imported. " + number.format(identical) +
        " identical. " + number.format(conflicts.length) + " to compare." +
        (!serverReviewMode && !reviewStore ? " Changes are available for this session." : "");
    } catch (error) {
      feedback.textContent = error instanceof SyntaxError ? "Choose a valid JSON backup." : error.message;
    } finally {
      restore.disabled = false;
    }
  };
  renderSavedReviews();
  if (serverReviewMode) {
    const refresh = () => requestServerReviews().catch(() => {});
    window.addEventListener("focus",refresh);
    id("saved-tab").addEventListener("click",refresh);
    refresh();
  } else {
    window.addEventListener("storage",event => {
      if (event.key === reviewStorageKey || event.key === null) updateReviewViews();
    });
  }
}
