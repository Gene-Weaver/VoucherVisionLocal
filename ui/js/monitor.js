// The resource monitor in the page header: processor, memory, and one bar per
// graphics card (GET /v1/metrics).
//
// It asks the server about every two seconds while the window is visible,
// waits longer after a failed request, and never shows an error for a missed
// reading: values it does not have are shown as "—".

import { api } from './api.js';
import { el, fmtMemory, replace } from './dom.js';
import { on } from './state.js';

const POLL_MS = 2000;
const BUSY_POLL_MS = 1000;      // while the engine is starting, loading or stopping
const MAX_WAIT_MS = 30000;
const REQUEST_TIMEOUT_MS = 6000;
const HIGH_PERCENT = 90;
const ENGINE_BUSY = new Set(['starting', 'loading_model', 'stopping']);
const DASH = '—';

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const pct = (v) => (isNum(v) ? `${Math.round(v)}%` : DASH);
const mem = (v, opts) => (isNum(v) ? fmtMemory(v, opts) : DASH);
const share = (part, total) => (isNum(part) && isNum(total) && total > 0 ? Math.max(0, Math.min(100, (part / total) * 100)) : 0);

/** "76.2 / 504 GB": the unit is said once when both numbers share it. */
function memPair(used, total, { short = false } = {}) {
    if (!isNum(used) && !isNum(total)) return DASH;
    const opts = { digits: short ? 0 : 1 };
    const u = mem(used, opts);
    const t = mem(total, opts);
    const unit = / (GB|MB)$/;
    if (unit.test(u) && unit.test(t) && u.slice(-2) === t.slice(-2)) return `${u.replace(unit, '')} / ${t}`;
    return `${u} / ${t}`;
}

function rows(list) {
    const dl = el('dl', { class: 'resmon-rows' });
    for (const row of list) {
        if (!row) continue;
        const [label, value, { key = null, sub = false, plain = false } = {}] = row;
        dl.append(
            el('dt', { class: sub ? 'is-sub' : null }, key ? el('span', { class: `resmon-key resmon-key-${key}`, 'aria-hidden': 'true' }) : null, label),
            el('dd', { class: plain ? 'is-plain' : null }, value));
    }
    return dl;
}

let seq = 0;

/** One monitor entry: a label, a value, a small bar, and a popover with the exact numbers. */
function item(kind, labelNodes) {
    const id = `resmon-pop-${seq += 1}`;
    const value = el('span', { class: 'resmon-value' }, DASH);
    const extra = el('span', { class: 'resmon-vram' });
    const warn = el('span', { class: 'resmon-warn', 'aria-hidden': 'true' }, '!');
    const bar = el('div', { class: 'resmon-bar', 'aria-hidden': 'true' });
    const box = el('div', { class: 'resmon-pop-box' });
    const pop = el('div', { class: 'resmon-pop', role: 'tooltip', id }, box);
    const node = el('div', { class: `resmon-item resmon-${kind}`, tabindex: '0', role: 'group', 'aria-describedby': id },
        el('div', { class: 'resmon-top' }, el('span', { class: 'resmon-label' }, labelNodes, warn), el('span', null, value, extra)),
        bar, pop);
    // Escape closes the popover until the pointer or the focus leaves.
    node.addEventListener('keydown', (ev) => {
        if (ev.key === 'Escape') node.classList.add('is-dismissed');
    });
    for (const name of ['mouseleave', 'blur']) node.addEventListener(name, () => node.classList.remove('is-dismissed'));
    const segs = {};
    const seg = (name) => {
        if (!segs[name]) {
            segs[name] = el('span', { class: `resmon-seg resmon-seg-${name}` });
            bar.appendChild(segs[name]);
        }
        return segs[name];
    };
    return { node, value, extra, box, seg };
}

function renderCpu(view, cpu) {
    const p = cpu && isNum(cpu.percent) ? cpu.percent : null;
    view.value.textContent = pct(p);
    const fill = view.seg('fill');
    fill.style.width = `${share(p, 100)}%`;
    fill.classList.toggle('is-high', p !== null && p >= HIGH_PERCENT);
    view.node.setAttribute('aria-label', `CPU: ${pct(p)}`);
    replace(view.box,
        el('div', { class: 'resmon-pop-title' }, 'CPU'),
        rows([
            ['Usage', pct(p)],
            cpu && isNum(cpu.logical_cores) ? ['Threads', String(cpu.logical_cores)] : null,
        ]));
}

function renderRam(view, memory, short) {
    const m = memory || {};
    view.value.textContent = memPair(m.used_bytes, m.total_bytes, { short });
    const fill = view.seg('fill');
    fill.style.width = `${share(m.used_bytes, m.total_bytes)}%`;
    fill.classList.toggle('is-high', isNum(m.percent) && m.percent >= HIGH_PERCENT);
    view.node.setAttribute('aria-label', `RAM: ${mem(m.used_bytes)} / ${mem(m.total_bytes)}`);
    replace(view.box,
        el('div', { class: 'resmon-pop-title' }, 'RAM'),
        rows([
            ['Used', `${memPair(m.used_bytes, m.total_bytes)}${isNum(m.percent) ? ` (${pct(m.percent)})` : ''}`],
            ['Free', mem(m.available_bytes)],
        ]));
}

function renderGpu(view, gpu, plan, short) {
    const total = gpu.vram_total_bytes;
    const used = gpu.vram_used_bytes;
    const knownSplit = isNum(gpu.engine_bytes) && isNum(gpu.other_bytes);
    const planned = isNum(gpu.planned_bytes) ? gpu.planned_bytes : 0;
    const free = isNum(gpu.vram_free_bytes) ? gpu.vram_free_bytes : null;
    const over = planned > 0 && gpu.planned_fits === false;

    view.value.textContent = pct(gpu.util_percent);
    view.extra.textContent = ` · ${memPair(used, total, { short })}`;
    view.node.classList.toggle('is-over', over);

    // Bar: the engine's memory, other programs' memory, then the planned part.
    const engineW = knownSplit ? share(gpu.engine_bytes, total) : 0;
    const otherW = knownSplit ? share(gpu.other_bytes, total) : share(used, total);
    const plannedW = Math.min(share(planned, total), Math.max(0, 100 - engineW - otherW));
    view.seg('engine').style.width = `${engineW}%`;
    view.seg('other').style.width = `${otherW}%`;
    // A planned amount that does not fit still shows as a sliver, in the warning color.
    view.seg('planned').style.width = `${planned > 0 ? Math.max(plannedW, isNum(total) ? 6 : 0) : 0}%`;

    const sharedMemory = gpu.memory_source === 'system_ram';
    const plannedFor = Array.isArray(gpu.planned_for) ? gpu.planned_for : [];
    let warning = '';
    if (over) {
        warning = `Needs ${mem(planned)} (${mem(free)} free). Close other apps or pick a smaller model.`;
    }
    const note = planned > 0 && plan && plan.mode === 'exclusive' && typeof plan.note === 'string' ? plan.note : '';

    view.node.setAttribute('aria-label',
        `GPU ${gpu.index}: ${pct(gpu.util_percent)}, ${mem(used)} / ${mem(total)}`
        + `${planned > 0 ? `, ${mem(planned)} planned` : ''}${over ? ', does not fit' : ''}`);
    replace(view.box,
        el('div', { class: 'resmon-pop-title' }, `GPU ${gpu.index}`),
        el('div', { class: 'resmon-pop-sub' }, gpu.name || 'GPU'),
        rows([
            ['Load', pct(gpu.util_percent)],
            ['Temp', isNum(gpu.temperature_c) ? `${Math.round(gpu.temperature_c)} °C` : DASH],
            [sharedMemory ? 'Shared memory' : 'Memory', memPair(used, total)],
            ['This app', mem(gpu.engine_bytes), { key: 'engine', sub: true }],
            ['Other apps', mem(gpu.other_bytes), { key: 'other', sub: true }],
            ['Selected models', planned > 0 ? mem(planned) : 'None', { key: 'planned', sub: true }],
            ...plannedFor.map((p) => [p.label || p.model_id, mem(p.bytes), { sub: true, plain: true }]),
            ['Free', mem(free), { key: 'free', sub: true }],
        ]),
        el('p', { class: 'resmon-pop-warn', role: 'note' }, warning),
        el('p', { class: 'resmon-pop-note' }, note));
}

/** Mount the monitor into `host`. Returns {refresh()}. */
export function mountMonitor(host) {
    if (!host) return { refresh() {} };
    const cpu = item('cpu', 'CPU');
    const ram = item('ram', 'RAM');
    const root = el('div', { class: 'resmon', role: 'group', 'aria-label': 'Computer resources in use' }, cpu.node, ram.node);
    replace(host, root);

    const narrow = window.matchMedia('(max-width: 1280px)');
    const gpuViews = new Map(); // card index -> view
    let last = null;
    let timer = null;
    let fetching = false;
    let again = false;
    let failures = 0;
    let stopped = false;

    function render() {
        const data = last || {};
        const short = narrow.matches;
        renderCpu(cpu, data.cpu);
        renderRam(ram, data.memory, short);
        const gpus = Array.isArray(data.gpus) ? data.gpus : [];
        const wanted = new Set(gpus.map((g) => g.index));
        for (const [index, view] of gpuViews) {
            if (!wanted.has(index)) {
                view.node.remove();
                gpuViews.delete(index);
            }
        }
        for (const gpu of gpus) {
            let view = gpuViews.get(gpu.index);
            if (!view) {
                view = item('gpu', [el('span', { class: 'resmon-label-word' }, 'GPU '), String(gpu.index)]);
                gpuViews.set(gpu.index, view);
                root.appendChild(view.node);
            }
            renderGpu(view, gpu, data.plan, short);
        }
    }

    function schedule(ms) {
        if (timer) clearTimeout(timer);
        timer = null;
        if (stopped || document.hidden) return;
        timer = setTimeout(tick, ms);
    }

    async function tick() {
        if (timer) clearTimeout(timer);
        timer = null;
        if (stopped || document.hidden) return;
        if (fetching) {
            again = true; // something changed while a request was on its way: ask once more after it
            return;
        }
        fetching = true;
        try {
            last = await api.get('/v1/metrics', { timeoutMs: REQUEST_TIMEOUT_MS, background: true });
            failures = 0;
            render();
        } catch (err) {
            failures += 1;
            if (err && err.status === 401) stopped = true; // the page shows its own message for this
        } finally {
            fetching = false;
        }
        if (again) {
            again = false;
            tick();
            return;
        }
        const busy = last && last.engine && ENGINE_BUSY.has(last.engine.state);
        schedule(failures ? Math.min(MAX_WAIT_MS, POLL_MS * (2 ** failures)) : (busy ? BUSY_POLL_MS : POLL_MS));
    }

    const refresh = () => {
        failures = 0;
        tick();
    };
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            if (timer) clearTimeout(timer);
            timer = null;
        } else {
            refresh();
        }
    });
    narrow.addEventListener('change', render);
    // The planned part follows the model choice and the engine, without waiting for the next reading.
    for (const name of ['selection-changed', 'engine-changed', 'models-changed', 'job-finished']) on(name, refresh);

    // For the desktop shell's screenshot run: the latest reading (no secrets in it).
    if (window.vvlUi) window.vvlUi.metrics = () => last;

    render();
    tick();
    return { refresh };
}
