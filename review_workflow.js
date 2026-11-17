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

function syncServerReviewForms() {
  for (const view of Object.values(reviewViews)) {
    id(view.inspector).querySelector(".review-entry form")?.dispatchEvent(new Event("reviewstorechange"));
  }
}

function updateReviewViews() {
  renderSavedReviews();
  paintQueueStatuses("award");
  paintQueueStatuses("tender");
  if (typeof refreshReviewStatusQueues === "function") refreshReviewStatusQueues();
  if (serverReviewMode) syncServerReviewForms();
}

function showReviewDatabaseUnavailable() {
  reviewConnection = "unavailable";
  updateReviewViews();
}

async function requestServerReviews(update) {
  let response, payload;
  try {
    response = update ? await fetch("/api/reviews",{
      method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(update)
    }) : await fetch("/api/reviews?snapshot_id=" + encodeURIComponent(data.snapshot_id),{cache:"no-store"});
    payload = await response.json();
  } catch {
    showReviewDatabaseUnavailable();
    throw new Error("Could not reach the app’s reviews database. Try again.");
  }
  if (!response.ok) {
    if (!update || response.status >= 500 || reviewRevision === null) showReviewDatabaseUnavailable();
    const error = new Error(typeof payload?.error === "string" ? payload.error : "The reviews database rejected the request.");
    error.status = response.status;
    throw error;
  }
  let entries;
  try {
    if (!Number.isSafeInteger(payload?.revision) || payload.revision < 0) throw new Error();
    entries = parseReviews(payload);
  } catch {
    showReviewDatabaseUnavailable();
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

function reviewNeedsEvidence(status) {
  return status === "reviewed" || status === "follow_up";
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
  const savedState = node("span","","review-entry-state");
  summary.append(node("span","Review case","review-entry-title")," ",savedState);
  const form = node("form");
  form.noValidate = true;
  const addInput = (target,input,label,name,full = false) => {
    const wrapper = node("label","",full ? "full" : "");
    const labelLine = node("span","","review-field-label");
    const title = node("span",label);
    title.id = kind + "-review-label-" + name;
    const required = node("span","Required","review-required");
    required.setAttribute("aria-hidden","true");
    required.hidden = true;
    labelLine.append(title," ",required);
    input.name = name;
    input.value = entered[name];
    input.setAttribute("aria-labelledby",title.id);
    if (name in reviewLimits) input.maxLength = reviewLimits[name];
    wrapper.append(labelLine,input);
    target.append(wrapper);
    return required;
  };
  const status = node("select");
  reviewStatuses.forEach((label,value) => option(status,label,value));
  addInput(form,status,"Review status","status");
  const reviewer = node("input");
  const reviewerRequired = addInput(form,reviewer,"Reviewer initials","reviewer");
  const statusHint = node("p","","small review-status-hint full");
  statusHint.id = kind + "-review-status-hint";
  status.setAttribute("aria-describedby",statusHint.id);
  form.append(statusHint);
  const evidenceGroup = node("fieldset","","review-evidence full");
  const evidenceLegend = node("legend","Evidence");
  const evidenceRequired = node("span","One required","review-required");
  evidenceLegend.append(" ",evidenceRequired);
  const evidenceFields = node("div","","review-evidence-fields");
  const evidence = node("input");
  evidence.type = "url";
  evidence.placeholder = "https://";
  addInput(evidenceFields,evidence,"HTTPS evidence link","evidence_url");
  const reference = node("input");
  reference.placeholder = "Document ID or file number";
  addInput(evidenceFields,reference,"Internal document reference","evidence_ref");
  evidenceGroup.append(evidenceLegend,evidenceFields);
  const notes = node("textarea");
  notes.rows = 3;
  const notesRequired = addInput(form,notes,"Notes","notes",true);
  form.append(evidenceGroup);
  const save = node("button","Save review","minor primary");
  save.type = "submit";
  const discard = node("button","Discard draft","minor");
  discard.type = "button";
  const actions = node("div","","actions full");
  const footer = node("div","","review-form-footer full");
  const pending = node("p","Unsaved draft. Save it before closing this page.","small review-draft-notice");
  const feedback = node("p","","small review-feedback full");
  feedback.id = kind + "-review-form-feedback";
  feedback.setAttribute("role","status");
  const connection = node("div","","review-database-notice");
  connection.hidden = true;
  const retry = node("button","Retry connection","minor");
  retry.type = "button";
  connection.append(node("p","Database unavailable. Drafts stay on this page.","small"),retry);
  actions.append(save,discard);
  footer.append(pending,connection,actions,feedback);
  form.append(footer);
  panel.append(summary,form);
  container.append(panel);

  const fieldNames = Object.keys(entered);
  const readFields = () => Object.fromEntries(fieldNames.map(name => [name,form.elements[name].value]));
  const syncRequirements = () => {
    const needsEvidence = reviewNeedsEvidence(status.value);
    reviewer.required = notes.required = needsEvidence;
    reviewerRequired.hidden = notesRequired.hidden = evidenceRequired.hidden = !needsEvidence;
    const hints = {
      in_review:"Log that you have started checking this case.",
      needs_documents:"Note which documents you still need.",
      reviewed:"Summarize what you verified and add an evidence link or internal document reference.",
      follow_up:"Record the next action and add an evidence link or internal document reference."
    };
    statusHint.textContent = hints[status.value];
  };
  const clearFieldErrors = () => {
    for (const field of [reviewer,notes,evidence,reference]) {
      field.removeAttribute("aria-invalid");
      field.removeAttribute("aria-errormessage");
    }
  };
  const showDraftState = dirty => {
    panel.dataset.reviewState = dirty ? "draft" : original ? "saved" : "not-started";
    savedState.dataset.state = panel.dataset.reviewState;
    savedState.textContent = dirty ? "Unsaved draft" : original ? "Saved · " + reviewStatuses.get(original.status) : "Not started";
    discard.hidden = pending.hidden = !dirty;
  };
  const resetForm = saved => {
    original = saved;
    const fields = reviewFields(saved);
    for (const name of fieldNames) form.elements[name].value = fields[name];
    setReviewDraft(key,null);
    clearFieldErrors();
    syncRequirements();
    showDraftState(false);
  };
  const paintFeedback = ({message = "",tone = ""} = {}) => {
    feedback.dataset.state = tone;
    if (feedback.textContent !== message) feedback.textContent = message;
  };
  const setSaveMessage = (message,tone = "") => {
    const result = {message,tone};
    if (serverReviewMode) {
      if (message) reviewSaveMessages.set(key,result);
      else reviewSaveMessages.delete(key);
    }
    paintFeedback(result);
  };
  let reconnecting = false;
  const syncServerForm = () => {
    const saving = reviewSaves.has(key);
    for (const control of form.elements) control.disabled = saving;
    connection.hidden = reviewConnection !== "unavailable";
    retry.disabled = saving || reconnecting;
    const saved = loadReviews().get(key);
    if (!saving && !reviewDrafts.has(key) && !sameReview(saved,original)) resetForm(saved);
    paintFeedback(reviewSaveMessages.get(key));
  };
  syncRequirements();
  showDraftState(Boolean(draft));
  if (serverReviewMode) {
    form.addEventListener("reviewstorechange",syncServerForm);
    syncServerForm();
    retry.onclick = async () => {
      reconnecting = true;
      setSaveMessage("Checking the app database…","pending");
      syncServerForm();
      try {
        await requestServerReviews();
        setSaveMessage("App database connection restored.","success");
        if (form.isConnected) save.focus({preventScroll:true});
      } catch (error) {
        setSaveMessage(error.message,"error");
      } finally {
        reconnecting = false;
        syncServerForm();
      }
    };
  }
  form.addEventListener("input",event => {
    const fields = readFields();
    const dirty = !sameReview(fields,reviewFields(original));
    setReviewDraft(key,dirty ? {original,fields} : null);
    showDraftState(dirty);
    if (event.target === status) syncRequirements();
    clearFieldErrors();
    setSaveMessage("");
  });
  discard.onclick = () => {
    resetForm(loadReviews().get(key));
    setSaveMessage("Draft discarded.","notice");
    notes.focus();
  };
  const showFieldError = (fields,message) => {
    for (const field of fields) {
      field.setAttribute("aria-invalid","true");
      field.setAttribute("aria-errormessage",feedback.id);
    }
    setSaveMessage(message,"error");
    fields[0].focus();
  };
  form.addEventListener("submit",async event => {
    event.preventDefault();
    if (reviewSaves.has(key)) return;
    clearFieldErrors();
    const fields = Object.fromEntries(Object.entries(readFields()).map(([name,value]) => [name,value.trim()]));
    if (evidenceLink(fields.evidence_url) === null) {
      showFieldError([evidence],"Use an HTTPS evidence link, or clear the link field.");
      return;
    }
    if (reviewNeedsEvidence(fields.status)) {
      const missing = new Map();
      if (!fields.reviewer) missing.set(reviewer,"your initials");
      if (!fields.notes) missing.set(notes,"notes explaining your decision");
      if (!fields.evidence_url && !fields.evidence_ref) missing.set(reference,"an evidence link or internal document reference");
      if (missing.size) {
        const labels = new Intl.ListFormat("en-PH").format(missing.values());
        showFieldError([...missing.keys()],"Enter " + labels + ".");
        return;
      }
    }
    if (serverReviewMode) {
      reviewSaves.add(key);
      setSaveMessage("Saving review…","pending");
      syncServerForm();
    }
    const warnIfCaseChanged = saved => {
      if (sameReview(saved,original)) return false;
      if (reviewDrafts.has(key)) {
        setSaveMessage("The saved review changed. Copy any draft text you need, then discard the draft to load the latest saved review.","error");
      } else {
        resetForm(saved);
        setSaveMessage("The saved review changed. The latest version is shown here.","notice");
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
        "Review saved in this browser." : "Review saved for this session. Download a backup in Saved reviews before closing.","success");
    } catch (error) {
      if (error.status === 409) {
        await requestServerReviews().catch(() => {});
        if (warnIfCaseChanged(loadReviews().get(key))) return;
      }
      setSaveMessage("Review could not be saved. " + error.message,"error");
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
  const statusFilter = id(kind + "ReviewStatus");
  if (statusFilter) statusFilter.value = "";
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
  if (typeof showReviewCase === "function") showReviewCase(kind);
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
  const appConnection = id("appConnection");
  if (appConnection) {
    const states = {
      connecting:["connecting","Connecting"],connected:["ready","Database ready"],
      unavailable:["unavailable","Database unavailable"]
    };
    const [state,message] = serverReviewMode ? states[reviewConnection] : reviewStore ?
      ["browser","Browser storage"] : ["session","Session only"];
    appConnection.dataset.state = state;
    if (appConnection.textContent !== message) appConnection.textContent = message;
  }
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
  const coverageDate = new Intl.DateTimeFormat("en-PH",{dateStyle:"medium",timeZone:"UTC"});
  const snapshotBanner = id("snapshotBanner");
  snapshotBanner.title = "Snapshot " + data.snapshot_id.slice(0,12);
  snapshotBanner.replaceChildren(node("strong","Historical data"),document.createTextNode(
    " · Tenders through " + coverageDate.format(new Date(snapshot.latest_tender_publication_date + "T00:00:00Z")) +
    " · Official awards through " + coverageDate.format(new Date(snapshot.latest_official_award_date + "T00:00:00Z")) +
    ". Confirm current deadlines in the official notice."
  ));
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
