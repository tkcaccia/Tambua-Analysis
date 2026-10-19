const resultRoot = 'assets/monusac-transfer/';
const imagePanels = [
  ['he', 'Original H&E', 'Expert-annotated prostate ROI'],
  ['expert', 'Expert cell types', 'Independent nuclear masks, colored by expert type'],
  ['predicted', 'Archived Tambua predictions', 'Pre-fix virtual markers; QC not validated'],
  ['errors', 'Correct / incorrect', 'Green nuclei correct; red nuclei incorrect'],
  ['dapi', 'Virtual DAPI — pre-fix', 'Archived marker preview with visible shadow'],
  ['ck', 'Virtual CK — pre-fix', 'Archived marker preview; QC not validated'],
  ['cd3', 'Virtual CD3 — pre-fix', 'Archived marker preview; QC not validated'],
  ['cd68', 'Virtual CD68 — pre-fix', 'Archived marker preview; QC not validated'],
];
function pct(value) { return (value * 100).toFixed(1) + '%'; }
function renderCase(sample) {
  const selected = window.monusacResults.cases.find(item => item.sample_id === sample);
  if (!selected) return;
  const gallery = document.getElementById('monusac-gallery');
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
  document.getElementById('monusac-case-meta').textContent =
    selected.sample_id + ' · patient ' + selected.patient_id + ' · ' + selected.width + ' × ' + selected.height +
    ' pixels · ' + selected.source_mpp + ' µm/pixel source · ' + selected.n_cells + ' expert cells · ' +
    selected.correct + ' correct (' + pct(selected.accuracy) + '). Virtual DAPI top-decile enrichment in expert nuclei: ' +
    selected.dapi_qc.top_decile_nuclear_enrichment.toFixed(2) + '×' +
    (selected.dapi_qc.passes_enrichment_screen ? '.' : ' — below the >1 screen.') +
    ' Median nucleus/local-ring signal: ' + selected.dapi_qc.median_nucleus_to_local_ring_ratio.toFixed(2) + '×; ' +
    pct(selected.dapi_qc.fraction_nuclei_brighter_than_local_ring) + ' of nuclei are brighter than their ring.';
  const body = document.getElementById('monusac-case-classes'); body.replaceChildren();
  for (const [index, name] of window.monusacResults.class_order.entries()) {
    const support = selected.support[name];
    const recall = support ? selected.confusion_matrix[index][index] / support : null;
    const row = document.createElement('tr');
    for (const value of [name, String(support), recall === null ? 'No reference cells' : pct(recall)]) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    }
    body.append(row);
  }
}
fetch(resultRoot + 'results.json').then(response => {
  if (!response.ok) throw Error('Result manifest unavailable'); return response.json();
}).then(data => {
  window.monusacResults = data;
  const select = document.getElementById('monusac-case');
  for (const [index, sample] of data.cases.entries()) {
    const option = document.createElement('option'); option.value = sample.sample_id;
    option.textContent = String(index + 1).padStart(2, '0') + ' · ' + sample.sample_id + ' · ' + pct(sample.accuracy);
    select.append(option);
  }
  select.addEventListener('change', () => renderCase(select.value));
  renderCase(data.cases[0].sample_id);
}).catch(error => { document.getElementById('monusac-case-meta').textContent = error.message; });
