// Viewer tab: inspect saved VoucherVision JSON results per project and per image.
// - Top: project picker + per-project aggregate stats (counts, costs, tokens, models)
// - Left: filterable image list with keyboard navigation
// - Middle: user-friendly pretty section views (default: Formatted + OCR; plus Cost & Tokens, Info & Status, OCR Details, Raw JSON)
// - Right: Input & Label Collage images stacked, decoded directly from JSON via independent processes (or Web Workers for local files), clickable for a zoomable/pannable popup.

import { api, plainError, toast, toastError } from '../api.js';
import {
    el, icon, infoTip, replace, clear, stateBox, badge, defList, copyText, openModal,
} from '../dom.js';
import { on } from '../state.js';
import { canPickFolders, openFolderButton, pickFolder } from '../output_folder.js';

const VIEWS = [
    { id: 'default', label: 'Formatted + OCR' },
    { id: 'cost', label: 'Cost & Tokens' },
    { id: 'info', label: 'Info & Status' },
    { id: 'ocr', label: 'OCR Details' },
    { id: 'raw', label: 'Raw JSON' },
];

// ---------------------------------------------------------------- formatting helpers

function fmtUsd(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n <= 0) return '$0.00';
    if (n < 0.01) return `$${n.toFixed(4)}`;
    if (n < 1) return `$${n.toFixed(3)}`;
    return `$${n.toFixed(2)}`;
}

function fmtInt(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '0';
    return Math.round(n).toLocaleString('en-US');
}

function parseFormattedJson(val) {
    if (val && typeof val === 'object' && !Array.isArray(val)) return val;
    if (typeof val === 'string' && val.trim()) {
        try {
            const parsed = JSON.parse(val);
            if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
        } catch (_) { /* leave null */ }
    }
    return null;
}

function truthyFlag(v) {
    if (typeof v === 'boolean') return v;
    return ['true', '1', 'yes'].includes(String(v || '').trim().toLowerCase());
}

function num(v) {
    if (v === null || v === undefined || v === '' || typeof v === 'boolean') return 0;
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
}

function computeClientMetrics(data) {
    const ocrInfo = (data && data.ocr_info && typeof data.ocr_info === 'object') ? data.ocr_info : {};
    let ocrIn = 0; let ocrOut = 0; let ocrThink = 0;
    let ocrCost = 0;
    const ocrEngines = [];
    const ocrModels = [];

    for (const [modelId, info] of Object.entries(ocrInfo)) {
        if (!info || typeof info !== 'object') continue;
        ocrModels.push(modelId);
        const tIn = Math.round(num(info.tokens_in ?? info.input));
        const tOut = Math.round(num(info.tokens_out ?? info.output));
        const tThk = Math.round(num(info.thinking_tokens));
        const cIn = num(info.cost_in);
        const cOut = num(info.cost_out);
        const cThk = num(info.thinking_cost);
        const cTot = info.total_cost !== undefined ? num(info.total_cost) : (cIn + cOut + cThk);
        ocrIn += tIn; ocrOut += tOut; ocrThink += tThk; ocrCost += cTot;
        ocrEngines.push({
            model: modelId, tokens_in: tIn, tokens_out: tOut, thinking_tokens: tThk,
            cost_in: cIn, cost_out: cOut, thinking_cost: cThk, total_cost: cTot,
        });
    }

    const p = (data && data.parsing_info && typeof data.parsing_info === 'object') ? data.parsing_info : {};
    const parseModel = String(p.model || '').trim();
    const parseIn = Math.round(num(p.input ?? p.tokens_in));
    const parseOut = Math.round(num(p.output ?? p.tokens_out));
    const parseThink = Math.round(num(p.thinking_tokens));
    const pCostIn = num(p.cost_in);
    const pCostOut = num(p.cost_out);
    const pCostThink = num(p.thinking_cost);
    const parseCost = p.total_cost !== undefined ? num(p.total_cost) : (pCostIn + pCostOut + pCostThink);

    const totalCost = (data && data.total_request_cost_usd !== undefined && data.total_request_cost_usd !== null && data.total_request_cost_usd !== '')
        ? num(data.total_request_cost_usd)
        : (ocrCost + parseCost);

    return {
        total_cost_usd: totalCost,
        ocr_cost_usd: ocrCost,
        parse_cost_usd: parseCost,
        total_tokens: ocrIn + ocrOut + ocrThink + parseIn + parseOut + parseThink,
        ocr_tokens_in: ocrIn,
        ocr_tokens_out: ocrOut,
        ocr_thinking_tokens: ocrThink,
        parse_tokens_in: parseIn,
        parse_tokens_out: parseOut,
        parse_thinking_tokens: parseThink,
        ocr_models: ocrModels,
        parse_model: parseModel,
        ocr_engines: ocrEngines,
        parsing: {
            model: parseModel,
            thinking_budget_requested: p.thinking_budget_requested || '',
            tokens_in: parseIn,
            tokens_out: parseOut,
            thinking_tokens: parseThink,
            cost_in: pCostIn,
            cost_out: pCostOut,
            thinking_cost: pCostThink,
            total_cost: parseCost,
        },
    };
}

function stripBase64Client(data) {
    const out = { ...(data || {}) };
    const collage = data && data.collage_info && typeof data.collage_info === 'object' ? data.collage_info : null;
    const rawImages = { input: null, collage: null };
    if (collage) {
        const clean = {};
        for (const [k, v] of Object.entries(collage)) {
            if (k === 'base64image_input_resized') {
                if (typeof v === 'string' && v.trim()) {
                    rawImages.input = v;
                    clean[k] = `<base64 image (${Math.round((v.length * 3) / 4096)} KB)>`;
                } else {
                    clean[k] = null;
                }
            } else if (k === 'base64image_text_collage') {
                if (typeof v === 'string' && v.trim()) {
                    rawImages.collage = v;
                    clean[k] = `<base64 image (${Math.round((v.length * 3) / 4096)} KB)>`;
                } else {
                    clean[k] = null;
                }
            } else {
                clean[k] = v;
            }
        }
        out.collage_info = clean;
    }
    return {
        record: out,
        rawImages,
        images: { has_input: Boolean(rawImages.input), has_collage: Boolean(rawImages.collage) },
    };
}

// ---------------------------------------------------------------- independent Web Worker pool (local files)

function createDecodeWorkerPool(size = 2) {
    let workers = [];
    let nextId = 0;
    let rr = 0;
    const pending = new Map();

    function ensureWorkers() {
        if (workers.length || typeof Worker !== 'function') return;
        for (let i = 0; i < size; i += 1) {
            const w = new Worker('/ui/js/viewer_worker.js');
            w.onmessage = (ev) => {
                const { id, ok, buffer, mime, error } = ev.data || {};
                const job = pending.get(id);
                if (!job) return;
                pending.delete(id);
                if (ok && buffer) job.resolve(new Blob([buffer], { type: mime || 'image/jpeg' }));
                else job.reject(new Error(error || 'Decode failed'));
            };
            w.onerror = () => {
                for (const [id, job] of pending.entries()) {
                    pending.delete(id);
                    job.reject(new Error('Worker error'));
                }
            };
            workers.push(w);
        }
    }

    function decode(base64) {
        ensureWorkers();
        if (!workers.length) {
            return Promise.reject(new Error('Workers unavailable'));
        }
        return new Promise((resolve, reject) => {
            const id = (nextId += 1);
            pending.set(id, { resolve, reject });
            const w = workers[rr % workers.length];
            rr += 1;
            w.postMessage({ id, base64 });
        });
    }

    function destroy() {
        for (const w of workers) w.terminate();
        workers = [];
        pending.clear();
    }

    return { decode, destroy };
}

// ---------------------------------------------------------------- zoomable image popup lightbox

function openImageLightbox({ initialKind, filename, images }) {
    let kind = initialKind;
    let scale = 1;
    let panX = 0;
    let panY = 0;
    let dragging = false;
    let dragStartX = 0;
    let dragStartY = 0;

    const titleEl = el('strong', { class: 'vlb-title' });
    const zoomLabel = el('span', { class: 'vlb-zoom-pct' }, '100%');
    const imgEl = el('img', { class: 'vlb-img', alt: '', draggable: 'false' });
    const stage = el('div', { class: 'vlb-stage' }, imgEl);

    const inputBtn = el('button', {
        type: 'button', class: 'vlb-seg-btn', disabled: !images.input,
        onclick: () => setKind('input'),
    }, 'Input');
    const collageBtn = el('button', {
        type: 'button', class: 'vlb-seg-btn', disabled: !images.collage,
        onclick: () => setKind('collage'),
    }, 'Collage');

    function applyTransform() {
        imgEl.style.setProperty('--vlb-scale', String(scale));
        imgEl.style.setProperty('--vlb-x', `${panX}px`);
        imgEl.style.setProperty('--vlb-y', `${panY}px`);
        stage.classList.toggle('is-zoomed', scale > 1.01);
        zoomLabel.textContent = `${Math.round(scale * 100)}%`;
    }

    function resetZoom(nextScale = 1) {
        scale = nextScale;
        panX = 0;
        panY = 0;
        applyTransform();
    }

    function zoomBy(factor) {
        const next = Math.max(0.25, Math.min(8, scale * factor));
        scale = next;
        if (scale <= 1) {
            panX = 0;
            panY = 0;
        }
        applyTransform();
    }

    function setKind(nextKind) {
        if (!images[nextKind]) return;
        kind = nextKind;
        inputBtn.setAttribute('aria-pressed', String(kind === 'input'));
        collageBtn.setAttribute('aria-pressed', String(kind === 'collage'));
        const label = kind === 'input' ? 'Input image' : 'Label collage';
        titleEl.textContent = `${label} — ${filename}`;
        imgEl.src = images[kind];
        imgEl.alt = `${label} for ${filename}`;
        resetZoom(1);
    }

    stage.addEventListener('wheel', (ev) => {
        ev.preventDefault();
        zoomBy(ev.deltaY < 0 ? 1.18 : 1 / 1.18);
    }, { passive: false });

    stage.addEventListener('dblclick', () => {
        resetZoom(scale > 1.2 ? 1 : 2.2);
    });

    stage.addEventListener('mousedown', (ev) => {
        if (ev.button !== 0 || scale <= 1.01) return;
        dragging = true;
        dragStartX = ev.clientX - panX;
        dragStartY = ev.clientY - panY;
        stage.classList.add('is-dragging');
    });

    function onMouseMove(ev) {
        if (!dragging) return;
        panX = ev.clientX - dragStartX;
        panY = ev.clientY - dragStartY;
        applyTransform();
    }

    function onMouseUp() {
        if (!dragging) return;
        dragging = false;
        stage.classList.remove('is-dragging');
    }

    function onKeyDown(ev) {
        if (ev.key === 'Escape') { ev.preventDefault(); close(); }
        else if (ev.key === '+' || ev.key === '=') { ev.preventDefault(); zoomBy(1.25); }
        else if (ev.key === '-') { ev.preventDefault(); zoomBy(1 / 1.25); }
        else if (ev.key === '0') { ev.preventDefault(); resetZoom(1); }
        else if (ev.key === 'ArrowLeft' && images.input) { ev.preventDefault(); setKind('input'); }
        else if (ev.key === 'ArrowRight' && images.collage) { ev.preventDefault(); setKind('collage'); }
    }

    const toolbar = el('div', { class: 'vlb-bar' },
        titleEl,
        el('div', { class: 'vlb-seg', role: 'group', 'aria-label': 'Image view' }, inputBtn, collageBtn),
        el('span', { class: 'spacer' }),
        el('div', { class: 'vlb-controls' },
            el('button', { type: 'button', class: 'vlb-btn', title: 'Zoom out (-)', 'aria-label': 'Zoom out', onclick: () => zoomBy(1 / 1.25) }, icon('zoomOut', { size: 16 })),
            zoomLabel,
            el('button', { type: 'button', class: 'vlb-btn', title: 'Zoom in (+)', 'aria-label': 'Zoom in', onclick: () => zoomBy(1.25) }, icon('zoomIn', { size: 16 })),
            el('button', { type: 'button', class: 'vlb-btn', title: 'Fit to window (0)', onclick: () => resetZoom(1) }, 'Fit'),
            el('button', { type: 'button', class: 'vlb-btn', title: '200% zoom', onclick: () => resetZoom(2) }, '2×'),
            el('button', { type: 'button', class: 'vlb-btn vlb-close', title: 'Close (Esc)', 'aria-label': 'Close', onclick: () => close() }, icon('x', { size: 18 }))));

    const backdrop = el('div', {
        class: 'vlb-backdrop', role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Image preview',
        onclick: (ev) => { if (ev.target === backdrop) close(); },
    }, toolbar, stage, el('div', { class: 'vlb-hint' }, 'Scroll to zoom · Drag to pan · Double-click to toggle zoom · Esc to close'));

    function close() {
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
        window.removeEventListener('keydown', onKeyDown);
        backdrop.remove();
    }

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    window.addEventListener('keydown', onKeyDown);
    document.body.appendChild(backdrop);
    setKind(kind);
}

// ---------------------------------------------------------------- main Viewer mount

export async function mount(panel) {
    let availableProjects = [];
    let mode = 'server';            // 'server' | 'local'
    let project = null;             // { id, name, path, json_dir }
    let stats = null;
    let records = [];               // index rows
    let localFiles = new Map();     // file -> parsed JSON (local browser mode only)
    let filterText = '';
    let selectedFile = null;
    let activeView = 'default';
    let hideEmpty = false;
    let currentDetail = null;       // { file, stem, record, images, metrics }
    let currentImageUrls = { input: null, collage: null };
    let recordSeq = 0;
    let imageAbort = null;
    const workerPool = createDecodeWorkerPool(2);

    function revokeImages() {
        for (const k of ['input', 'collage']) {
            if (currentImageUrls[k]) {
                URL.revokeObjectURL(currentImageUrls[k]);
                currentImageUrls[k] = null;
            }
        }
    }

    // ------------------------------------------------------------ DOM structure
    const projectSelect = el('select', {
        id: 'viewerProjectSelect', class: 'viewer-project-select', 'aria-label': 'Project folder',
        onchange: () => {
            if (projectSelect.value) openServerProject({ path: projectSelect.value });
        },
    });
    const refreshBtn = el('button', {
        type: 'button', class: 'button-secondary button-small', title: 'Refresh runs',
        onclick: () => loadProjectList({ keepOpen: true }),
    }, icon('refresh', { size: 13 }), 'Refresh');
    const browseBtn = canPickFolders()
        ? el('button', {
            type: 'button', class: 'button-secondary button-small',
            onclick: async () => {
                try {
                    const picked = await pickFolder();
                    if (picked) await openServerProject({ path: picked });
                } catch (_) {
                    toast('Could not open folder picker.', 'error');
                }
            },
        }, icon('folder', { size: 13 }), 'Browse folder')
        : null;
    const localInput = el('input', { type: 'file', accept: '.json', multiple: true, class: 'visually-hidden', id: 'viewerLocalJsonInput' });
    const localBtn = el('button', {
        type: 'button', class: 'button-secondary button-small',
        onclick: () => localInput.click(),
    }, icon('file', { size: 13 }), 'Load JSON');
    const openDirHost = el('span', { class: 'viewer-open-host' });

    // The project's totals live in a popup, opened from the "Project Stats" button beside the project selector.
    const statsHost = el('div', { class: 'viewer-stats-host' });
    const statsBtn = el('button', {
        type: 'button', class: 'button-secondary button-small viewer-stats-btn', hidden: true,
        onclick: () => openModal({ title: 'Project Stats', body: statsHost, className: 'viewer-stats-modal' }),
    }, 'Project Stats');
    const topBar = el('div', { class: 'viewer-topbar' },
        el('label', { for: projectSelect.id, class: 'viewer-bar-label' }, 'Project'),
        projectSelect,
        statsBtn,
        refreshBtn,
        browseBtn,
        localBtn,
        localInput,
        el('span', { class: 'spacer' }),
        openDirHost);

    const filterInput = el('input', {
        type: 'text', class: 'viewer-filter', placeholder: 'Filter images…', 'aria-label': 'Filter images',
        oninput: () => { filterText = filterInput.value; renderRecordList(); },
    });
    const prevBtn = el('button', {
        type: 'button', class: 'pd-mini-btn', title: 'Previous image', 'aria-label': 'Previous image',
        onclick: () => stepRecord(-1),
    }, icon('chevronLeft', { size: 14 }));
    const nextBtn = el('button', {
        type: 'button', class: 'pd-mini-btn', title: 'Next image', 'aria-label': 'Next image',
        onclick: () => stepRecord(1),
    }, icon('chevronRight', { size: 14 }));
    const listCount = el('span', { class: 'viewer-list-count' });
    const listHost = el('div', { class: 'viewer-list', role: 'listbox', 'aria-label': 'Images in project' });
    const middleHost = el('div', { class: 'viewer-middle' });
    const imagesHost = el('div', { class: 'viewer-images' });

    const bodyGrid = el('div', { class: 'viewer-grid', hidden: true },
        el('aside', { class: 'viewer-sidebar' },
            el('div', { class: 'viewer-sidebar-head' }, filterInput, el('div', { class: 'viewer-nav-row' }, listCount, el('span', { class: 'spacer' }), prevBtn, nextBtn)),
            listHost),
        middleHost,
        imagesHost);
    const emptyHost = el('div', { class: 'viewer-empty-host' });

    replace(panel, el('div', { class: 'viewer-shell' }, topBar, emptyHost, bodyGrid));

    // ------------------------------------------------------------ stats strip
    function renderStats() {
        statsBtn.hidden = !stats || !project;
        if (!stats || !project) {
            clear(statsHost);
            return;
        }
        const total = stats.total_records || 0;
        const avgCost = total ? (stats.total_cost_usd || 0) / total : 0;
        const avgTok = total ? (stats.total_tokens || 0) / total : 0;
        const inTok = (stats.ocr_tokens_in || 0) + (stats.parse_tokens_in || 0);
        const outTok = (stats.ocr_tokens_out || 0) + (stats.parse_tokens_out || 0);
        const thinkTok = (stats.ocr_thinking_tokens || 0) + (stats.parse_thinking_tokens || 0);
        const modelsList = [...new Set([...(stats.ocr_models || []), ...(stats.parse_models || [])])];
        const promptsList = stats.prompts || [];

        const statCard = (label, value, sub = null) => el('div', { class: 'viewer-stat' },
            el('div', { class: 'viewer-stat-label' }, label),
            el('div', { class: 'viewer-stat-val' }, value),
            sub ? el('div', { class: 'viewer-stat-sub', title: sub }, sub) : null);

        replace(statsHost, el('div', { class: 'viewer-stats' },
            statCard('Images', fmtInt(total), `${fmtInt(stats.parsed_records || 0)} parsed${stats.has_coords ? ` · ${fmtInt(stats.has_coords)} geolocated` : ''}`),
            statCard('Total cost', fmtUsd(stats.total_cost_usd), `OCR ${fmtUsd(stats.ocr_cost_usd)} · Parse ${fmtUsd(stats.parse_cost_usd)}`),
            statCard('Total tokens', fmtInt(stats.total_tokens), `In ${fmtInt(inTok)} · Out ${fmtInt(outTok)}${thinkTok ? ` · Think ${fmtInt(thinkTok)}` : ''}`),
            statCard('Avg / image', fmtUsd(avgCost), `${fmtInt(avgTok)} tokens`),
            statCard('Models & prompt',
                promptsList.length ? promptsList.join(', ') : (modelsList[0] || '—'),
                modelsList.length ? modelsList.join(' · ') : null)));
    }

    // ------------------------------------------------------------ record list
    function filteredRecords() {
        const q = filterText.trim().toLowerCase();
        if (!q) return records;
        return records.filter((r) => `${r.stem} ${r.filename} ${r.scientific_name || ''} ${r.catalog_number || ''}`.toLowerCase().includes(q));
    }

    function stepRecord(delta) {
        const list = filteredRecords();
        if (!list.length) return;
        const idx = list.findIndex((r) => r.file === selectedFile);
        const nextIdx = idx < 0 ? 0 : Math.max(0, Math.min(list.length - 1, idx + delta));
        if (list[nextIdx] && list[nextIdx].file !== selectedFile) selectRecord(list[nextIdx].file);
    }

    function renderRecordList() {
        const list = filteredRecords();
        listCount.textContent = `${list.length} of ${records.length}`;
        const idx = list.findIndex((r) => r.file === selectedFile);
        prevBtn.disabled = idx <= 0;
        nextBtn.disabled = idx < 0 || idx >= list.length - 1;

        if (!list.length) {
            replace(listHost, el('p', { class: 'help-text viewer-list-empty' }, records.length ? 'No matching images.' : 'No JSON files in this folder.'));
            return;
        }
        replace(listHost, list.map((r) => {
            const onRow = r.file === selectedFile;
            const sub = [r.scientific_name, r.catalog_number].filter(Boolean).join(' · ');
            return el('button', {
                type: 'button', role: 'option',
                class: ['viewer-rec-item', onRow ? 'selected' : ''],
                'aria-selected': String(onRow),
                onclick: () => selectRecord(r.file),
            },
            el('div', { class: 'viewer-rec-top' },
                el('span', { class: 'viewer-rec-name', title: r.filename }, r.stem),
                r.cost_usd > 0 ? el('span', { class: 'viewer-rec-cost' }, fmtUsd(r.cost_usd)) : null),
            sub ? el('div', { class: 'viewer-rec-sub', title: sub }, sub) : null);
        }));
    }

    // ------------------------------------------------------------ middle views
    function formattedFieldsTable(obj) {
        const entries = Object.entries(obj);
        const visible = hideEmpty
            ? entries.filter(([, v]) => v !== null && v !== undefined && String(v).trim() !== '')
            : entries;
        if (!visible.length) return el('p', { class: 'help-text' }, hideEmpty ? 'All fields are empty.' : 'No fields.');
        return el('table', { class: 'field-table viewer-field-table' },
            el('tbody', null, visible.map(([k, v]) => {
                const empty = v === null || v === undefined || v === '';
                const text = empty ? '—' : (typeof v === 'object' ? JSON.stringify(v) : String(v));
                return el('tr', null,
                    el('th', { scope: 'row' }, k),
                    el('td', { class: empty ? 'field-empty' : null }, text));
            })));
    }

    function renderDefaultView(rec) {
        const fj = parseFormattedJson(rec.formatted_json);
        const md = typeof rec.formatted_md === 'string' && rec.formatted_md.trim() ? rec.formatted_md : '';
        const ocr = typeof rec.ocr === 'string' ? rec.ocr : (rec.ocr ? JSON.stringify(rec.ocr, null, 2) : '');

        const hideEmptyBox = el('input', {
            type: 'checkbox', id: 'viewerHideEmpty', checked: hideEmpty,
            onchange: () => { hideEmpty = hideEmptyBox.checked; renderMiddle(); },
        });

        const formattedCard = el('div', { class: 'card viewer-card' },
            el('div', { class: 'viewer-card-head' },
                el('h3', null, 'Formatted JSON'),
                fj ? badge(`${Object.keys(fj).length} fields`, 'neutral') : null,
                el('span', { class: 'spacer' }),
                fj ? el('label', { class: 'check-row viewer-inline-check', for: 'viewerHideEmpty' }, hideEmptyBox, 'Hide empty') : null,
                fj ? el('button', {
                    type: 'button', class: 'button-secondary button-small',
                    onclick: async () => toast((await copyText(JSON.stringify(fj, null, 2))) ? 'Copied.' : 'Copy failed.', 'info'),
                }, icon('copy', { size: 12 }), 'Copy') : null),
            fj ? formattedFieldsTable(fj)
                : (md ? el('pre', { class: 'viewer-ocr-pre' }, md) : el('p', { class: 'help-text' }, 'No formatted fields in this record.')));

        const ocrCard = el('div', { class: 'card viewer-card' },
            el('div', { class: 'viewer-card-head' },
                el('h3', null, 'Final OCR'),
                el('span', { class: 'spacer' }),
                ocr.trim() ? el('button', {
                    type: 'button', class: 'button-secondary button-small',
                    onclick: async () => toast((await copyText(ocr)) ? 'Copied.' : 'Copy failed.', 'info'),
                }, icon('copy', { size: 12 }), 'Copy') : null),
            ocr.trim() ? el('pre', { class: 'viewer-ocr-pre' }, ocr) : el('p', { class: 'help-text' }, 'No OCR text.'));

        return [formattedCard, ocrCard];
    }

    function renderCostView(metrics) {
        const m = metrics || {};
        const p = m.parsing || {};
        const engines = m.ocr_engines || [];

        const summary = el('div', { class: 'viewer-cost-strip' },
            el('div', { class: 'viewer-stat' }, el('div', { class: 'viewer-stat-label' }, 'Request total'), el('div', { class: 'viewer-stat-val' }, fmtUsd(m.total_cost_usd))),
            el('div', { class: 'viewer-stat' }, el('div', { class: 'viewer-stat-label' }, 'OCR cost'), el('div', { class: 'viewer-stat-val' }, fmtUsd(m.ocr_cost_usd))),
            el('div', { class: 'viewer-stat' }, el('div', { class: 'viewer-stat-label' }, 'Parsing cost'), el('div', { class: 'viewer-stat-val' }, fmtUsd(m.parse_cost_usd))),
            el('div', { class: 'viewer-stat' }, el('div', { class: 'viewer-stat-label' }, 'Total tokens'), el('div', { class: 'viewer-stat-val' }, fmtInt(m.total_tokens))));

        const ocrTable = engines.length
            ? el('div', { class: 'table-wrap' }, el('table', { class: 'data' },
                el('thead', null, el('tr', null,
                    el('th', null, 'OCR model'), el('th', null, 'In'), el('th', null, 'Out'), el('th', null, 'Think'),
                    el('th', null, 'Cost in'), el('th', null, 'Cost out'), el('th', null, 'Total'))),
                el('tbody', null, engines.map((e) => el('tr', null,
                    el('td', null, e.model),
                    el('td', null, fmtInt(e.tokens_in)),
                    el('td', null, fmtInt(e.tokens_out)),
                    el('td', null, fmtInt(e.thinking_tokens)),
                    el('td', null, fmtUsd(e.cost_in)),
                    el('td', null, fmtUsd(e.cost_out)),
                    el('td', null, el('strong', null, fmtUsd(e.total_cost))))))))
            : el('p', { class: 'help-text' }, 'No OCR cost data.');

        const parseRows = [
            ['Model', p.model || '—'],
            p.thinking_budget_requested ? ['Thinking budget', String(p.thinking_budget_requested)] : null,
            ['Input tokens', fmtInt(p.tokens_in)],
            ['Output tokens', fmtInt(p.tokens_out)],
            ['Thinking tokens', fmtInt(p.thinking_tokens)],
            ['Input cost', fmtUsd(p.cost_in)],
            ['Output cost', fmtUsd(p.cost_out)],
            ['Thinking cost', fmtUsd(p.thinking_cost)],
            ['Total parsing cost', fmtUsd(p.total_cost)],
        ].filter(Boolean);

        return [
            summary,
            el('div', { class: 'card viewer-card' }, el('div', { class: 'viewer-card-head' }, el('h3', null, 'OCR engines')), ocrTable),
            el('div', { class: 'card viewer-card' }, el('div', { class: 'viewer-card-head' }, el('h3', null, 'Parsing')), defList(parseRows)),
        ];
    }

    function renderInfoView(rec) {
        const success = rec.success && typeof rec.success === 'object' ? rec.success : {};
        const infoRows = [
            ['Filename', rec.filename || '—'],
            ['Prompt', rec.prompt || '—'],
            ['Host', rec.host || '—'],
            rec.url_source ? ['Source URL', rec.url_source] : null,
            rec.engine_notice ? ['Engine notice', rec.engine_notice] : null,
        ].filter(Boolean);

        const statusBadges = Object.entries(success).map(([k, v]) => {
            const valStr = String(v);
            const isBool = ['true', 'false'].includes(valStr.toLowerCase());
            return el('span', { class: 'viewer-status-pill' },
                el('span', { class: 'viewer-status-key' }, k),
                badge(valStr, isBool ? (truthyFlag(v) ? 'good' : 'neutral') : 'info'));
        });

        const enrichCards = [];
        if (rec.COP90_elevation_m !== undefined && rec.COP90_elevation_m !== null && rec.COP90_elevation_m !== '') {
            enrichCards.push(el('div', { class: 'card viewer-card' },
                el('div', { class: 'viewer-card-head' }, el('h3', null, 'Elevation (COP90)')),
                el('p', null, el('strong', null, `${rec.COP90_elevation_m} m`))));
        }
        if (rec.WFO_info && typeof rec.WFO_info === 'object' && Object.keys(rec.WFO_info).length) {
            enrichCards.push(el('div', { class: 'card viewer-card' },
                el('div', { class: 'viewer-card-head' }, el('h3', null, 'World Flora Online')),
                formattedFieldsTable(rec.WFO_info)));
        }

        const collageMeta = {};
        if (rec.collage_info && typeof rec.collage_info === 'object') {
            for (const [k, v] of Object.entries(rec.collage_info)) {
                if (k !== 'base64image_input_resized' && k !== 'base64image_text_collage') collageMeta[k] = v;
            }
        }

        return [
            el('div', { class: 'card viewer-card' },
                el('div', { class: 'viewer-card-head' }, el('h3', null, 'Run info')),
                defList(infoRows)),
            statusBadges.length ? el('div', { class: 'card viewer-card' },
                el('div', { class: 'viewer-card-head' }, el('h3', null, 'Pipeline status')),
                el('div', { class: 'viewer-status-grid' }, statusBadges)) : null,
            ...enrichCards,
            Object.keys(collageMeta).length ? el('div', { class: 'card viewer-card' },
                el('div', { class: 'viewer-card-head' }, el('h3', null, 'Collage metadata')),
                formattedFieldsTable(collageMeta)) : null,
        ].filter(Boolean);
    }

    function renderOcrDetailsView(rec) {
        const ocrInfo = rec.ocr_info && typeof rec.ocr_info === 'object' ? rec.ocr_info : {};
        const cards = Object.entries(ocrInfo).map(([modelId, info]) => {
            if (!info || typeof info !== 'object') return null;
            const text = typeof info.text === 'string' ? info.text : (typeof info.ocr === 'string' ? info.ocr : '');
            const meta = Object.entries(info).filter(([k]) => k !== 'text' && k !== 'ocr');
            return el('div', { class: 'card viewer-card' },
                el('div', { class: 'viewer-card-head' },
                    el('h3', null, modelId),
                    info.total_cost !== undefined ? badge(fmtUsd(info.total_cost), 'neutral') : null),
                meta.length ? defList(meta.map(([k, v]) => [k, typeof v === 'object' ? JSON.stringify(v) : String(v)])) : null,
                text ? el('pre', { class: 'viewer-ocr-pre' }, text) : null);
        }).filter(Boolean);

        const ocr = typeof rec.ocr === 'string' ? rec.ocr : '';
        return [
            ...cards,
            el('div', { class: 'card viewer-card' },
                el('div', { class: 'viewer-card-head' }, el('h3', null, 'Combined final OCR')),
                ocr.trim() ? el('pre', { class: 'viewer-ocr-pre' }, ocr) : el('p', { class: 'help-text' }, 'No OCR text.')),
        ];
    }

    function renderRawView(rec) {
        const pretty = JSON.stringify(rec, null, 2);
        return el('div', { class: 'card viewer-card' },
            el('div', { class: 'viewer-card-head' },
                el('h3', null, 'Raw JSON'),
                infoTip('Base64 image payloads are abbreviated here to keep display fast.', 'start'),
                el('span', { class: 'spacer' }),
                el('button', {
                    type: 'button', class: 'button-secondary button-small',
                    onclick: async () => toast((await copyText(pretty)) ? 'Copied.' : 'Copy failed.', 'info'),
                }, icon('copy', { size: 12 }), 'Copy')),
            el('pre', { class: 'json-content viewer-raw-pre' }, pretty));
    }

    function renderMiddle() {
        if (!currentDetail) {
            replace(middleHost, stateBox('empty', 'Select an image'));
            return;
        }
        const rec = currentDetail.record || {};
        const header = el('div', { class: 'viewer-mid-head' },
            el('div', { class: 'viewer-mid-title-row' },
                el('h2', { class: 'viewer-rec-heading', title: rec.filename || currentDetail.file }, rec.filename || currentDetail.stem),
                currentDetail.metrics && currentDetail.metrics.total_cost_usd > 0
                    ? badge(fmtUsd(currentDetail.metrics.total_cost_usd), 'neutral') : null,
                currentDetail.metrics && currentDetail.metrics.total_tokens > 0
                    ? badge(`${fmtInt(currentDetail.metrics.total_tokens)} tok`, 'neutral') : null),
            el('div', { class: 'viewer-view-tabs', role: 'tablist', 'aria-label': 'JSON section view' },
                VIEWS.map((v) => el('button', {
                    type: 'button', role: 'tab', class: 'viewer-view-btn',
                    'aria-selected': String(activeView === v.id),
                    onclick: () => { activeView = v.id; renderMiddle(); },
                }, v.label))));

        let content;
        if (activeView === 'default') content = renderDefaultView(rec);
        else if (activeView === 'cost') content = renderCostView(currentDetail.metrics);
        else if (activeView === 'info') content = renderInfoView(rec);
        else if (activeView === 'ocr') content = renderOcrDetailsView(rec);
        else content = renderRawView(rec);

        replace(middleHost, header, el('div', { class: 'viewer-mid-body' }, content));
    }

    // ------------------------------------------------------------ stacked right images (independent decoding)
    function makeImageCard(kind, title) {
        const body = el('div', { class: 'viewer-img-box' });
        const expandBtn = el('button', {
            type: 'button', class: 'pd-mini-btn', title: 'Open large zoom view', 'aria-label': `Zoom ${title}`, hidden: true,
            onclick: () => openPopup(kind),
        }, icon('maximize', { size: 13 }));
        const card = el('div', { class: 'card viewer-img-card' },
            el('div', { class: 'viewer-img-head' },
                el('span', { class: 'viewer-img-title' }, title),
                el('span', { class: 'spacer' }),
                expandBtn),
            body);
        return { card, body, expandBtn };
    }

    const inputCard = makeImageCard('input', 'Input image');
    const collageCard = makeImageCard('collage', 'Label collage');
    replace(imagesHost, collageCard.card, inputCard.card);

    // ---- spotlight: a magnified 16:9 window that follows the cursor over an image
    const SPOT_W = 480;
    const SPOT_H = 270;
    const SPOT_GAP = 25;       // px between the cursor and the window's bottom edge
    const SPOT_MARGIN = 8;
    const spotlight = el('div', { class: 'viewer-spotlight', hidden: true, 'aria-hidden': 'true' });
    document.body.appendChild(spotlight);
    function hideSpotlight() {
        spotlight.hidden = true;
    }
    function moveSpotlight(ev, img) {
        const box = img.getBoundingClientRect();
        if (!img.naturalWidth || !box.width || !box.height
            || ev.clientX < box.left || ev.clientX > box.right || ev.clientY < box.top || ev.clientY > box.bottom) {
            hideSpotlight();
            return;
        }
        const w = Math.min(SPOT_W, window.innerWidth - 2 * SPOT_MARGIN);
        const h = Math.round((w * 9) / 16);
        // Full resolution when the image has it, and always at least 3x what is on screen.
        const zoom = Math.max(3, Math.min(8, img.naturalWidth / box.width));
        const bgW = box.width * zoom;
        const bgH = box.height * zoom;
        const fx = (ev.clientX - box.left) / box.width;
        const fy = (ev.clientY - box.top) / box.height;
        // Above the cursor; below it when there is no room; never past a screen edge.
        let top = ev.clientY - SPOT_GAP - h;
        if (top < SPOT_MARGIN) top = ev.clientY + SPOT_GAP;
        top = Math.max(SPOT_MARGIN, Math.min(window.innerHeight - h - SPOT_MARGIN, top));
        const left = Math.max(SPOT_MARGIN, Math.min(window.innerWidth - w - SPOT_MARGIN, ev.clientX - w / 2));
        // Set through the style object: the page's security policy refuses style="" attributes.
        spotlight.style.width = `${w}px`;
        spotlight.style.height = `${h}px`;
        spotlight.style.left = `${Math.round(left)}px`;
        spotlight.style.top = `${Math.round(top)}px`;
        spotlight.style.backgroundImage = `url("${img.src}")`;
        spotlight.style.backgroundSize = `${bgW}px ${bgH}px`;
        spotlight.style.backgroundPosition = `${Math.round(w / 2 - fx * bgW)}px ${Math.round(h / 2 - fy * bgH)}px`;
        spotlight.hidden = false;
    }
    function attachSpotlight(target, img) {
        target.addEventListener('mousemove', (ev) => moveSpotlight(ev, img));
        target.addEventListener('mouseleave', hideSpotlight);
        target.addEventListener('click', hideSpotlight);
    }
    window.addEventListener('scroll', hideSpotlight, true);
    window.addEventListener('blur', hideSpotlight);

    function openPopup(kind) {
        if (!currentImageUrls[kind]) return;
        const fname = (currentDetail && currentDetail.record && currentDetail.record.filename) || (currentDetail && currentDetail.file) || '';
        openImageLightbox({
            initialKind: kind,
            filename: fname,
            images: { input: currentImageUrls.input, collage: currentImageUrls.collage },
        });
    }

    function setCardState(cardObj, state, urlOrText = null, kind = null) {
        cardObj.expandBtn.hidden = state !== 'ready';
        if (state === 'loading') {
            replace(cardObj.body, el('div', { class: 'viewer-img-placeholder' }, el('span', { class: 'spinner spinner-small', 'aria-hidden': 'true' }), el('span', null, 'Decoding…')));
        } else if (state === 'ready' && urlOrText) {
            const img = el('img', {
                src: urlOrText,
                alt: kind === 'input' ? 'Input image' : 'Label collage',
                class: 'viewer-thumb',
            });
            const thumb = el('button', {
                type: 'button', class: 'viewer-thumb-btn', 'aria-label': 'Open zoomable view',
                onclick: () => openPopup(kind),
            }, img, el('span', { class: 'viewer-thumb-hint' }, icon('zoomIn', { size: 14 }), 'Zoom'));
            attachSpotlight(thumb, img);
            replace(cardObj.body, thumb);
        } else {
            replace(cardObj.body, el('div', { class: 'viewer-img-placeholder' }, icon('image', { size: 20 }), el('span', null, urlOrText || 'Not included in JSON')));
        }
    }

    async function loadImagesIndependently(seq, file, imageMeta, rawLocalImages = null) {
        if (imageAbort) imageAbort.abort();
        imageAbort = new AbortController();
        const { signal } = imageAbort;
        revokeImages();

        const pairs = [
            { kind: 'input', card: inputCard, has: Boolean(imageMeta && imageMeta.has_input), raw: rawLocalImages && rawLocalImages.input },
            { kind: 'collage', card: collageCard, has: Boolean(imageMeta && imageMeta.has_collage), raw: rawLocalImages && rawLocalImages.collage },
        ];

        // Launch both decodes concurrently as independent tasks/processes.
        await Promise.all(pairs.map(async ({ kind, card, has, raw }) => {
            if (!has) {
                setCardState(card, 'empty', 'Not in JSON');
                return;
            }
            setCardState(card, 'loading');
            try {
                let blob;
                if (mode === 'server') {
                    blob = await api.blob(
                        `/v1/viewer/projects/${encodeURIComponent(project.id)}/images/${encodeURIComponent(file)}/${kind}`,
                        { signal, timeoutMs: 60000 },
                    );
                } else {
                    blob = await workerPool.decode(raw);
                }
                if (seq !== recordSeq || signal.aborted) return;
                const url = URL.createObjectURL(blob);
                currentImageUrls[kind] = url;
                setCardState(card, 'ready', url, kind);
            } catch (err) {
                if (seq !== recordSeq || (err && err.aborted)) return;
                setCardState(card, 'empty', 'Could not decode image');
            }
        }));
    }

    // ------------------------------------------------------------ selection & project loading
    async function selectRecord(file) {
        selectedFile = file;
        renderRecordList();
        const seq = (recordSeq += 1);

        if (mode === 'local') {
            const rawObj = localFiles.get(file);
            if (!rawObj) return;
            const { record, rawImages, images } = stripBase64Client(rawObj);
            currentDetail = {
                file,
                stem: file.replace(/\.json$/i, ''),
                record,
                images,
                metrics: computeClientMetrics(rawObj),
            };
            renderMiddle();
            loadImagesIndependently(seq, file, images, rawImages);
            return;
        }

        replace(middleHost, stateBox('loading', `Loading ${file}…`));
        setCardState(inputCard, 'loading');
        setCardState(collageCard, 'loading');
        try {
            const detail = await api.get(`/v1/viewer/projects/${encodeURIComponent(project.id)}/records/${encodeURIComponent(file)}`);
            if (seq !== recordSeq) return;
            currentDetail = detail;
            renderMiddle();
            loadImagesIndependently(seq, file, detail.images);
        } catch (err) {
            if (seq !== recordSeq) return;
            replace(middleHost, stateBox('error', `Could not load ${file}`, plainError(err)));
            setCardState(inputCard, 'empty', 'Unavailable');
            setCardState(collageCard, 'empty', 'Unavailable');
        }
    }

    function renderProjectDropdown() {
        const selectedVal = mode === 'server' && project ? project.path : '';
        const options = [];
        if (!availableProjects.length && !selectedVal) {
            options.push(el('option', { value: '' }, 'No saved runs yet'));
        } else {
            options.push(el('option', { value: '' }, 'Select a run…'));
            for (const p of availableProjects) {
                options.push(el('option', { value: p.path }, `${p.name} (${p.json_count})`));
            }
            if (selectedVal && !availableProjects.some((p) => p.path === selectedVal)) {
                options.push(el('option', { value: selectedVal }, `${project.name} (${records.length})`));
            }
        }
        replace(projectSelect, options);
        projectSelect.value = selectedVal;
    }

    async function openServerProject({ path, folder }) {
        clear(emptyHost);
        bodyGrid.hidden = true;
        replace(emptyHost, stateBox('loading', 'Opening project…'));
        try {
            const res = await api.post('/v1/viewer/open', path ? { path } : { folder });
            mode = 'server';
            localFiles.clear();
            project = res.project;
            stats = res.stats;
            records = res.records || [];
            selectedFile = null;
            currentDetail = null;
            filterText = '';
            filterInput.value = '';
            revokeImages();
            renderProjectDropdown();
            replace(openDirHost, openFolderButton(project.path, 'Open folder'));
            renderStats();
            if (!records.length) {
                bodyGrid.hidden = true;
                replace(emptyHost, stateBox('empty', 'No JSON files found in this folder'));
                return;
            }
            clear(emptyHost);
            bodyGrid.hidden = false;
            renderRecordList();
            await selectRecord(records[0].file);
        } catch (err) {
            bodyGrid.hidden = true;
            replace(emptyHost, stateBox('error', 'Could not open project', plainError(err)));
            toastError(err, 'Could not open project');
        }
    }

    // Load local JSON files directly (decoded via independent Web Workers)
    localInput.addEventListener('change', async () => {
        const files = [...(localInput.files || [])].filter((f) => f.name.toLowerCase().endsWith('.json'));
        localInput.value = '';
        if (!files.length) return;
        clear(emptyHost);
        bodyGrid.hidden = true;
        replace(emptyHost, stateBox('loading', `Reading ${files.length} file${files.length === 1 ? '' : 's'}…`));

        const parsedMap = new Map();
        const agg = {
            total_records: 0, parsed_records: 0, ocr_ok: 0, llm_ok: 0, collage_ok: 0,
            has_coords: 0, has_elevation: 0, has_wfo: 0,
            total_cost_usd: 0, ocr_cost_usd: 0, parse_cost_usd: 0,
            total_tokens: 0, ocr_tokens_in: 0, ocr_tokens_out: 0, ocr_thinking_tokens: 0,
            parse_tokens_in: 0, parse_tokens_out: 0, parse_thinking_tokens: 0,
            ocr_models: [], parse_models: [], prompts: [],
        };
        const ocrSet = new Set(); const parseSet = new Set(); const promptSet = new Set();
        const idxRows = [];

        for (const f of files.sort((a, b) => a.name.localeCompare(b.name))) {
            try {
                const obj = JSON.parse(await f.text());
                if (!obj || typeof obj !== 'object' || Array.isArray(obj)) continue;
                parsedMap.set(f.name, obj);
                const m = computeClientMetrics(obj);
                const fj = parseFormattedJson(obj.formatted_json) || {};
                const hasCoords = Number.isFinite(parseFloat(fj.decimalLatitude)) && Number.isFinite(parseFloat(fj.decimalLongitude));
                agg.total_records += 1;
                if (Object.keys(fj).length) agg.parsed_records += 1;
                if (hasCoords) agg.has_coords += 1;
                agg.total_cost_usd += m.total_cost_usd;
                agg.ocr_cost_usd += m.ocr_cost_usd;
                agg.parse_cost_usd += m.parse_cost_usd;
                agg.total_tokens += m.total_tokens;
                agg.ocr_tokens_in += m.ocr_tokens_in;
                agg.ocr_tokens_out += m.ocr_tokens_out;
                agg.ocr_thinking_tokens += m.ocr_thinking_tokens;
                agg.parse_tokens_in += m.parse_tokens_in;
                agg.parse_tokens_out += m.parse_tokens_out;
                agg.parse_thinking_tokens += m.parse_thinking_tokens;
                for (const om of m.ocr_models) ocrSet.add(om);
                if (m.parse_model) parseSet.add(m.parse_model);
                if (obj.prompt) promptSet.add(String(obj.prompt));

                const c = obj.collage_info || {};
                idxRows.push({
                    file: f.name,
                    stem: f.name.replace(/\.json$/i, ''),
                    filename: String(obj.filename || f.name),
                    scientific_name: String(fj.scientificName || '').trim(),
                    catalog_number: String(fj.catalogNumber || '').trim(),
                    cost_usd: m.total_cost_usd,
                    total_tokens: m.total_tokens,
                    has_input_image: typeof c.base64image_input_resized === 'string' && Boolean(c.base64image_input_resized.trim()),
                    has_collage_image: typeof c.base64image_text_collage === 'string' && Boolean(c.base64image_text_collage.trim()),
                });
            } catch (_) { /* skip invalid json */ }
        }

        if (!idxRows.length) {
            replace(emptyHost, stateBox('error', 'No valid JSON records found'));
            return;
        }
        agg.ocr_models = [...ocrSet].sort();
        agg.parse_models = [...parseSet].sort();
        agg.prompts = [...promptSet].sort();

        mode = 'local';
        localFiles = parsedMap;
        project = { id: 'local', name: `${idxRows.length} loaded file${idxRows.length === 1 ? '' : 's'}`, path: '' };
        stats = agg;
        records = idxRows;
        selectedFile = null;
        currentDetail = null;
        filterText = '';
        filterInput.value = '';
        revokeImages();
        clear(openDirHost);
        renderProjectDropdown();
        renderStats();
        clear(emptyHost);
        bodyGrid.hidden = false;
        renderRecordList();
        await selectRecord(records[0].file);
    });

    async function loadProjectList({ keepOpen = false } = {}) {
        try {
            const data = await api.get('/v1/viewer/projects');
            availableProjects = (data && data.projects) || [];
            renderProjectDropdown();
            if (!keepOpen && !project) {
                if (availableProjects.length) {
                    await openServerProject({ path: availableProjects[0].path });
                } else {
                    bodyGrid.hidden = true;
                    clear(statsHost);
                    replace(emptyHost, stateBox('empty', 'No saved runs yet', 'Process images on the VVL tab, or browse a folder of JSON results above.'));
                }
            }
        } catch (err) {
            if (!project) replace(emptyHost, stateBox('error', 'Could not list runs', plainError(err)));
        }
    }

    // Keyboard navigation when focused inside the sidebar list
    listHost.addEventListener('keydown', (ev) => {
        if (ev.key === 'ArrowDown') { ev.preventDefault(); stepRecord(1); }
        else if (ev.key === 'ArrowUp') { ev.preventDefault(); stepRecord(-1); }
    });

    on('output-changed', () => loadProjectList({ keepOpen: true }));

    panel.dataset.state = 'loading';
    await loadProjectList();
    panel.dataset.state = 'ready';

    return {
        async show(target) {
            if (target) {
                await loadProjectList({ keepOpen: true });
                await openServerProject({ path: target });
            } else {
                await loadProjectList({ keepOpen: Boolean(project) });
            }
        },
    };
}
