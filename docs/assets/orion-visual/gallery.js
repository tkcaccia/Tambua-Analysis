"use strict";

async function initializeOrionGallery() {
  const root = "assets/orion-visual/";
  const patientSelect = document.getElementById("orion-patient");
  const markerSelect = document.getElementById("orion-marker");
  const image = document.getElementById("orion-image");
  const link = document.getElementById("orion-image-link");
  const caption = document.getElementById("orion-caption");
  if (!patientSelect || !markerSelect || !image || !link || !caption) return;

  try {
    const response = await fetch(root + "manifest.json");
    if (!response.ok) throw new Error("Manifest HTTP " + response.status);
    const manifest = await response.json();
    const rows = manifest.records;
    if (!Array.isArray(rows) || rows.length !== 100) throw new Error("Incomplete comparison manifest");
    const patients = [...new Set(rows.map(row => row.patient))];
    const markers = [...new Set(rows.map(row => row.marker))];
    if (patients.length !== 10 || markers.length !== 10) throw new Error("Unexpected patient or marker count");
    patientSelect.replaceChildren(...patients.map(patient => new Option(patient, patient)));
    markerSelect.replaceChildren(...markers.map(marker => {
      const row = rows.find(record => record.marker === marker);
      return new Option(`${marker} → ${row.measured}`, marker);
    }));
    patientSelect.value = "CRC04";
    markerSelect.value = "CK";

    function update() {
      const row = rows.find(record => record.patient === patientSelect.value && record.marker === markerSelect.value);
      if (!row) throw new Error("Comparison image missing");
      const src = root + row.image;
      image.src = src;
      image.alt = `${row.patient} H&E, measured ${row.measured}, original GigaTIME ${row.marker}, and selected ${row.method} spatial cell-level maps`;
      link.href = src;
      const sign = value => `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
      caption.textContent = `${row.patient} / ${row.marker} → ${row.measured}. Original rho ${sign(row.baseline_rho)}; selected ${row.method.replaceAll("_", " ")} rho ${sign(row.selected_rho)}; paired change ${sign(row.delta_rho)} across ${row.common_cells.toLocaleString()} matched cells. Open the image for full resolution.`;
    }
    patientSelect.addEventListener("change", update);
    markerSelect.addEventListener("change", update);
    update();
  } catch (error) {
    caption.textContent = `The image selector could not load: ${error.message}. The default CRC04/CK comparison remains visible.`;
  }
}

document.addEventListener("DOMContentLoaded", initializeOrionGallery);
