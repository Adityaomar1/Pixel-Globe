// COSMOS Change Detection - Frontend Application Engine

let activeModelId = 'model_1';
let currentOverlayColor = '#ff0055';
let imageFiles = { t1: null, t2: null };
let imageBase64 = { t1: null, t2: null };
let currentResults = null;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    fetchSystemInfo();
    setupDropzones();
    setupSwipeSlider();
    setupClipboardPaste();
});

// 1. Fetch System Telemetry & Available Models
async function fetchSystemInfo() {
    try {
        const res = await fetch('/api/info');
        const data = await res.json();

        // Update hardware badge
        const devBadge = document.getElementById('device-badge');
        if (devBadge) {
            devBadge.innerHTML = `⚡ HARDWARE: <b>${data.device_name || data.device}</b>`;
        }

        // Update model details if available
        if (data.models && data.models.length > 0) {
            data.models.forEach(m => {
                if (m.id === 'model_1') {
                    document.getElementById('m1-name').innerText = m.name;
                    document.getElementById('m1-badge').innerText = m.badge;
                    document.getElementById('m1-backbone').innerText = m.backbone;
                } else if (m.id === 'model_2') {
                    document.getElementById('m2-name').innerText = m.name;
                    document.getElementById('m2-badge').innerText = m.badge;
                    document.getElementById('m2-backbone').innerText = m.backbone;
                }
            });
            activeModelId = data.active_model_id || 'model_1';
            updateModelCardUI();
        }

        // Render Quick Sample Chips
        const sampleContainer = document.getElementById('samples-container');
        if (sampleContainer && data.samples && data.samples.length > 0) {
            sampleContainer.innerHTML = '';
            data.samples.forEach(s => {
                const chip = document.createElement('button');
                chip.className = 'sample-chip';
                chip.innerHTML = `🛰️ ${s.name}`;
                chip.onclick = () => loadSample(s.t1_url, s.t2_url);
                sampleContainer.appendChild(chip);
            });
        }
    } catch (err) {
        console.warn('Could not connect to backend telemetry:', err);
    }
}

// 2. Model Switching
async function selectModel(modelId) {
    if (activeModelId === modelId) return;
    activeModelId = modelId;
    updateModelCardUI();

    try {
        const res = await fetch('/api/switch_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId })
        });
        const data = await res.json();
        if (data.success) {
            console.log(`Model switched to ${modelId}`);
        }
    } catch (err) {
        console.error('Failed to switch model on server:', err);
    }
}

function updateModelCardUI() {
    document.querySelectorAll('.model-card').forEach(card => card.classList.remove('active'));
    const activeCard = document.getElementById(`card-${activeModelId}`);
    if (activeCard) activeCard.classList.add('active');
}

// 3. File Selection & Dropzones
function setupDropzones() {
    ['t1', 't2'].forEach(type => {
        const dropzone = document.getElementById(`dropzone-${type}`);
        if (!dropzone) return;

        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.remove('drag-over');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length > 0) {
                processImageFile(files[0], type);
            }
        });
    });
}

function triggerFileSelect(type) {
    document.getElementById(`file-input-${type}`).click();
}

function handleFileSelect(event, type) {
    const file = event.target.files[0];
    if (file) {
        processImageFile(file, type);
    }
}

function processImageFile(file, type) {
    imageFiles[type] = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        const b64 = e.target.result;
        imageBase64[type] = b64;
        displayImagePreview(b64, type);
    };
    reader.readAsDataURL(file);
}

function displayImagePreview(src, type) {
    const imgEl = document.getElementById(`preview-${type}`);
    const emptyEl = document.getElementById(`empty-${type}`);
    const metaEl = document.getElementById(`meta-${type}`);

    imgEl.src = src;
    imgEl.style.display = 'block';
    emptyEl.style.display = 'none';

    // Get image natural dimensions
    const img = new Image();
    img.onload = () => {
        metaEl.innerText = `${img.naturalWidth} × ${img.naturalHeight}`;
        metaEl.style.display = 'block';
    };
    img.src = src;
}

function clearImage(type) {
    imageFiles[type] = null;
    imageBase64[type] = null;
    const imgEl = document.getElementById(`preview-${type}`);
    const emptyEl = document.getElementById(`empty-${type}`);
    const metaEl = document.getElementById(`meta-${type}`);
    const inputEl = document.getElementById(`file-input-${type}`);

    imgEl.src = '';
    imgEl.style.display = 'none';
    emptyEl.style.display = 'flex';
    metaEl.style.display = 'none';
    if (inputEl) inputEl.value = '';
}

function swapImages() {
    const tempFile = imageFiles.t1;
    imageFiles.t1 = imageFiles.t2;
    imageFiles.t2 = tempFile;

    const tempB64 = imageBase64.t1;
    imageBase64.t1 = imageBase64.t2;
    imageBase64.t2 = tempB64;

    if (imageBase64.t1) displayImagePreview(imageBase64.t1, 't1');
    else clearImage('t1');

    if (imageBase64.t2) displayImagePreview(imageBase64.t2, 't2');
    else clearImage('t2');
}

// 4. Clipboard Paste Support
function setupClipboardPaste() {
    window.addEventListener('paste', (e) => {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let item of items) {
            if (item.type.indexOf('image') === 0) {
                const blob = item.getAsFile();
                if (!imageBase64.t1) {
                    processImageFile(blob, 't1');
                } else if (!imageBase64.t2) {
                    processImageFile(blob, 't2');
                }
                break;
            }
        }
    });
}

// 5. Load Sample Orbit
async function loadSample(t1Url, t2Url) {
    try {
        const r1 = await fetch(t1Url);
        const b1 = await r1.blob();
        processImageFile(b1, 't1');

        const r2 = await fetch(t2Url);
        const b2 = await r2.blob();
        processImageFile(b2, 't2');
    } catch (err) {
        console.error('Error loading sample pair:', err);
    }
}

// 6. UI Controls (Threshold, Alpha, Color)
function updateThresholdLabel(val) {
    document.getElementById('threshold-val').innerText = parseFloat(val).toFixed(2);
}

function updateAlphaLabel(val) {
    document.getElementById('alpha-val').innerText = `${Math.round(val * 100)}%`;
}

function selectColor(el, colorHex) {
    document.querySelectorAll('.color-option').forEach(opt => opt.classList.remove('selected'));
    el.classList.add('selected');
    currentOverlayColor = colorHex;
}

// 7. Execute Inference
async function executeInference() {
    if (!imageBase64.t1 || !imageBase64.t2) {
        alert('⚠️ Please load both [T1 Before] and [T2 After] images before engaging detection.');
        return;
    }

    const loader = document.getElementById('loading-overlay');
    loader.style.display = 'flex';
    if (window.setHyperspace) window.setHyperspace(true);

    const threshold = document.getElementById('threshold-slider').value;
    const alpha = document.getElementById('alpha-slider').value;

    const formData = new FormData();
    if (imageFiles.t1) formData.append('image_t1', imageFiles.t1);
    else formData.append('image_t1', imageBase64.t1);

    if (imageFiles.t2) formData.append('image_t2', imageFiles.t2);
    else formData.append('image_t2', imageBase64.t2);

    formData.append('model_id', activeModelId);
    formData.append('threshold', threshold);
    formData.append('overlay_color', currentOverlayColor);
    formData.append('alpha', alpha);

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        const json = await res.json();
        loader.style.display = 'none';
        if (window.setHyperspace) window.setHyperspace(false);

        if (json.success && json.data) {
            renderResults(json.data);
        } else {
            alert('❌ Detection error: ' + (json.error || 'Unknown error'));
        }
    } catch (err) {
        loader.style.display = 'none';
        if (window.setHyperspace) window.setHyperspace(false);
        alert('❌ Network / Server Error: ' + err.message);
    }
}

// 8. Render Results & Telemetry
function renderResults(data) {
    currentResults = data;
    const resultsSection = document.getElementById('results-deck');
    resultsSection.style.display = 'block';

    // Telemetry stats
    const stats = data.stats;
    document.getElementById('stat-changed-pct').innerText = `${stats.changed_percentage}%`;
    document.getElementById('stat-changed-pixels').innerText = `${stats.changed_pixels.toLocaleString()} px`;
    document.getElementById('stat-latency').innerText = `${stats.latency_ms} ms`;
    
    const anomalyEl = document.getElementById('stat-anomaly');
    anomalyEl.innerText = stats.anomaly_level;
    anomalyEl.className = 'tele-value';
    if (stats.anomaly_level === 'CRITICAL') anomalyEl.classList.add('anomaly-critical');
    else if (stats.anomaly_level === 'MODERATE' || stats.anomaly_level === 'ELEVATED') anomalyEl.classList.add('anomaly-elevated');
    else anomalyEl.classList.add('anomaly-low');

    // Images
    document.getElementById('swipe-t1-img').src = imageBase64.t1;
    document.getElementById('swipe-overlay-img').src = data.overlay_base64;
    document.getElementById('img-result-overlay').src = data.overlay_base64;
    document.getElementById('img-result-mask').src = data.mask_base64;
    document.getElementById('img-result-heatmap').src = data.heatmap_base64;
    document.getElementById('img-result-composite').src = data.composite_base64;

    // Reset swipe handle to center
    resetSwipeSlider();

    // Default to swipe view
    switchViewMode('swipe');

    // Smooth scroll down to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// 9. Interactive Swipe Comparison Slider
function setupSwipeSlider() {
    const container = document.getElementById('swipe-container');
    const topLayer = document.getElementById('swipe-top-layer');
    const handle = document.getElementById('swipe-handle');
    let isDragging = false;

    function moveSlider(clientX) {
        const rect = container.getBoundingClientRect();
        let posX = clientX - rect.left;
        if (posX < 0) posX = 0;
        if (posX > rect.width) posX = rect.width;

        const pct = (posX / rect.width) * 100;
        topLayer.style.width = `${pct}%`;
        handle.style.left = `${pct}%`;

        // Sync top layer image width
        const topImg = document.getElementById('swipe-t1-img');
        if (topImg) {
            topImg.style.width = `${rect.width}px`;
            topImg.style.height = `${rect.height}px`;
        }
    }

    handle.addEventListener('mousedown', () => (isDragging = true));
    window.addEventListener('mouseup', () => (isDragging = false));
    window.addEventListener('mousemove', (e) => {
        if (isDragging) moveSlider(e.clientX);
    });

    // Touch Support
    handle.addEventListener('touchstart', () => (isDragging = true));
    window.addEventListener('touchend', () => (isDragging = false));
    window.addEventListener('touchmove', (e) => {
        if (isDragging && e.touches.length > 0) moveSlider(e.touches[0].clientX);
    });
}

function resetSwipeSlider() {
    const topLayer = document.getElementById('swipe-top-layer');
    const handle = document.getElementById('swipe-handle');
    const container = document.getElementById('swipe-container');
    const topImg = document.getElementById('swipe-t1-img');

    topLayer.style.width = '50%';
    handle.style.left = '50%';

    if (container && topImg) {
        const rect = container.getBoundingClientRect();
        topImg.style.width = `${rect.width}px`;
        topImg.style.height = `${rect.height}px`;
    }
}

// 10. Tab View Modes
function switchViewMode(mode) {
    document.querySelectorAll('.view-tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.single-view-container').forEach(c => (c.style.display = 'none'));
    const swipeContainer = document.getElementById('swipe-container');

    const activeBtn = Array.from(document.querySelectorAll('.view-tab-btn')).find(b =>
        b.getAttribute('onclick').includes(mode)
    );
    if (activeBtn) activeBtn.classList.add('active');

    if (mode === 'swipe') {
        swipeContainer.style.display = 'block';
        resetSwipeSlider();
    } else {
        swipeContainer.style.display = 'none';
        const target = document.getElementById(`view-${mode}`);
        if (target) target.style.display = 'flex';
    }
}

// 11. Download Artifacts
function downloadCurrent(type) {
    if (!currentResults) return;
    let b64 = '';
    let fname = 'change_detection';

    if (type === 'overlay') {
        b64 = currentResults.overlay_base64;
        fname = 'orbital_change_overlay.png';
    } else if (type === 'mask') {
        b64 = currentResults.mask_base64;
        fname = 'anomaly_binary_mask.png';
    } else if (type === 'heatmap') {
        b64 = currentResults.heatmap_base64;
        fname = 'cosmic_probability_heatmap.png';
    } else if (type === 'composite') {
        b64 = currentResults.composite_base64;
        fname = 'synoptic_3panel_report.png';
    }

    if (!b64) return;
    const a = document.createElement('a');
    a.href = b64;
    a.download = fname;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}
