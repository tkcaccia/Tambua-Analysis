(() => {
const resultRoot = 'assets/monusac-matched/';
const imagePanels = [
  ['he', 'Original H&E', 'Expert-annotated held-out ROI'],
  ['expert', 'Expert cell types', 'Independent nuclear masks, colored by expert type'],
  ['predicted', 'Archived Tambua predictions', 'Pre-fix virtual markers; QC failed; gray means unscored'],
  ['errors', 'Correct / incorrect', 'Green correct; red incorrect; gray unscored'],
  ['dapi', 'Virtual DAPI — pre-fix', 'Archived marker preview with visible shadow; QC failed'],
  ['ck', 'Virtual CK — pre-fix', 'Archived marker preview; QC failed'],
  ['cd3', 'Virtual CD3 — pre-fix', 'Archived marker preview; QC failed'],
  ['cd68', 'Virtual CD68 — pre-fix', 'Archived marker preview; QC failed'],
];
function pct(value) { return (value * 100).toFixed(1) + '%'; }
function renderCase(sample) {
  const selected = window.matchedResults.cases.find(item => item.sample_id === sample);
  if (!selected) return;
  const coverage = window.coverageById.get(sample);
  const nuclearQc = window.legacyNuclearQc.find(item => item.sample_id === sample);
  const gallery = document.getElementById('matched-gallery');
  gallery.replaceChildren();
  for (const [key, title, note] of imagePanels) {
    const figure = document.createElement('figure'); figure.className = 'image-card';
    const img = document.createElement('img');
    img.src = resultRoot + selected.sample_id + '-' + key + '.webp';
    img.alt = title + ' for ' + selected.sample_id; img.loading = 'lazy';
    const caption = document.createElement('figcaption');
    const strong = document.createElement('strong'); strong.textContent = title;
    const small = document.createElement('small'); small.textContent = note;
    caption.append(strong, small); figure.append(img, caption); gallery.append(figure);
  }
  document.getElementById('matched-case-meta').textContent =
    selected.organ + ' · ' + selected.sample_id + ' · patient ' + selected.patient_id + ' · ' + selected.width + ' × ' + selected.height +
    ' pixels · ' + selected.source_mpp + ' µm/pixel source · ' + selected.n_cells + ' expert cells · ' +
    coverage.reference_class_count + ' expert cell type' + (coverage.reference_class_count === 1 ? '' : 's') +
    ' (' + coverage.reference_classes.join(', ') + '). ' +
    (coverage.multiclass_eligible ? 'Within-image mean class recall: ' + pct(coverage.observed_class_macro_recall) +
      '; macro F1: ' + pct(coverage.observed_class_macro_f1) + '. ' :
      'Single-class image: multiclass discrimination is not measurable. ') +
    selected.correct + ' / ' + selected.n_cells + ' historical classifications correct (' + pct(selected.accuracy) + '); ' + selected.annotated_cells + ' expert nuclei total. Virtual DAPI top-decile enrichment in expert nuclei: ' +
    selected.dapi_qc.top_decile_nuclear_enrichment.toFixed(2) + '×' +
    (selected.dapi_qc.passes_enrichment_screen ? '.' : ' — below the >1 screen.') +
    ' Median nucleus/local-ring signal: ' + selected.dapi_qc.median_nucleus_to_local_ring_ratio.toFixed(2) + '×; ' +
    pct(selected.dapi_qc.fraction_nuclei_brighter_than_local_ring) + ' of nuclei are brighter than their ring. ' +
    (nuclearQc ? 'Retrospective QC on the saved native-size, pre-fix virtual DAPI JPEG versus expert nuclei: AUC ' + nuclearQc.auc.toFixed(3) +
      ', Dice ' + nuclearQc.prevalence_matched_dice.toFixed(3) + ', boundary contrast ' + (nuclearQc.nuclear_boundary_contrast === null ? 'insufficient' : nuclearQc.nuclear_boundary_contrast.toFixed(3)) + ', status ' + nuclearQc.status.toUpperCase() + '. These markers are provisional pending corrected inference.' :
      'Retrospective nuclear QC unavailable for this case.');
  const body = document.getElementById('matched-case-classes'); body.replaceChildren();
  for (const [index, name] of window.matchedResults.class_order.entries()) {
    const support = selected.support[name];
    const recall = support ? selected.confusion_matrix[index][index] / support : null;
    const row = document.createElement('tr');
    for (const value of [name, String(support), recall === null ? 'No reference cells' : pct(recall)]) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    }
    body.append(row);
  }
}
Promise.all(['results.json', 'nuclear-qc-legacy-boundary-v2.json'].map(file => fetch(resultRoot + file).then(response => {
  if (!response.ok) throw Error(file + ' unavailable'); return response.json();
}))).then(([data, qc]) => {
  window.matchedResults = data;
  window.legacyNuclearQc = qc;
  window.coverageById = new Map(data.class_coverage.per_image.map(item => [item.sample_id, item]));
  const select = document.getElementById('matched-case');
  const multiclassOnly = document.getElementById('matched-multiclass-only');
  for (const [index, sample] of data.cases.entries()) {
    const option = document.createElement('option'); option.value = sample.sample_id;
    const coverage = window.coverageById.get(sample.sample_id);
    option.textContent = String(index + 1).padStart(2, '0') + ' · ' + sample.organ + ' · ' + sample.sample_id + ' · ' +
      (coverage.multiclass_eligible ? coverage.reference_class_count + ' classes · mean recall ' + pct(coverage.observed_class_macro_recall) :
        'ONE CLASS · no multiclass score');
    select.append(option);
  }
  multiclassOnly.addEventListener('change', () => {
    for (const option of select.options) option.hidden = multiclassOnly.checked && !window.coverageById.get(option.value).multiclass_eligible;
    if (select.selectedOptions[0]?.hidden) {
      select.value = data.cases.find(item => window.coverageById.get(item.sample_id).multiclass_eligible).sample_id;
      renderCase(select.value);
    }
  });
  select.addEventListener('change', () => renderCase(select.value));
  renderCase(data.cases[0].sample_id);
}).catch(error => { document.getElementById('matched-case-meta').textContent = error.message; });
})();
