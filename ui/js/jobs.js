// Background jobs: model downloads and engine installs (GET /v1/jobs).
//
// One poller for the whole page. It asks the server about once a second, and
// only while a job is active. Tabs listen for two events on the shared bus:
//   jobs-changed   the latest list of jobs
//   job-finished   one job that just went from active to done / error / canceled

import { api } from './api.js';
import { el, fmtBytes, fmtDuration, replace } from './dom.js';
import { emit, gotoTab, on } from './state.js';

const ACTIVE_STATES = new Set(['queued', 'running', 'verifying']);
const POLL_MS = 1000;
const RETRY_MS = 3000;

let jobs = [];
let timer = null;
let fetching = null;
const wasActive = new Set();

export function isActive(job) {
    return Boolean(job && ACTIVE_STATES.has(job.state));
}

export function allJobs() {
    return jobs;
}

export function getJob(id) {
    return id ? jobs.find((j) => j.id === id) || null : null;
}

/** The running job of one kind ('model_download' | 'engine_install' | 'detector_download'), else a queued one, else null. */
export function activeJob(kind) {
    const mine = jobs.filter((j) => j.kind === kind && isActive(j));
    return mine.find((j) => j.state !== 'queued') || mine[0] || null;
}

/** The newest job for a build or engine build: the active one first, else the last one listed. */
export function jobFor(kind, target) {
    const mine = jobs.filter((j) => j.kind === kind && j.target === target);
    return mine.find(isActive) || mine[mine.length - 1] || null;
}

function apply(list) {
    jobs = Array.isArray(list) ? list : [];
    const finished = [];
    for (const job of jobs) {
        if (isActive(job)) wasActive.add(job.id);
        else if (wasActive.delete(job.id)) finished.push(job);
    }
    emit('jobs-changed', jobs);
    for (const job of finished) emit('job-finished', job);
}

function schedule(ms) {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
        timer = null;
        refreshJobs();
    }, ms);
}

/** Ask the server for the job list now. Keeps polling while anything is active. */
export function refreshJobs() {
    if (fetching) return fetching;
    fetching = api.get('/v1/jobs', { timeoutMs: 15000 }).then((data) => {
        fetching = null;
        apply(data && data.jobs);
        if (jobs.some(isActive)) schedule(POLL_MS);
        return jobs;
    }, () => {
        fetching = null;
        if (wasActive.size) schedule(RETRY_MS); // keep watching a job we know about
        return jobs;
    });
    return fetching;
}

/** Record a job the server just returned (a started download or install) and start watching it. */
export function trackJob(job) {
    if (!job || !job.id) return;
    const i = jobs.findIndex((j) => j.id === job.id);
    const next = jobs.slice();
    if (i >= 0) next[i] = job;
    else next.push(job);
    apply(next);
    schedule(POLL_MS);
}

// ---------------------------------------------------------------- display helpers
export function jobPercent(job) {
    if (!job || !(job.bytes_total > 0)) return 0;
    return Math.max(0, Math.min(100, Math.floor((job.bytes_done / job.bytes_total) * 100)));
}

const ERROR_HINTS = {
    NETWORK_ERROR: 'Check the internet connection, then try again. The download continues from where it stopped.',
    DOWNLOAD_FAILED: 'Try again. The download continues from where it stopped.',
    HTTP_ERROR: 'The download server refused the request. Try again later.',
    INSUFFICIENT_DISK_SPACE: 'Free some disk space or choose another models folder on the Engine tab, then try again.',
    GATED_MODEL: 'This model needs a Hugging Face token and its license accepted on the model page. Add the token on the Keys tab, then try again.',
    EXTRACT_FAILED: 'The downloaded engine could not be unpacked. Try again.',
};

/** {message, hint} for a failed job; both are plain sentences. */
export function jobError(job) {
    const err = (job && job.error) || {};
    return {
        message: err.message || 'The job stopped with an error.',
        hint: ERROR_HINTS[err.code] || null,
        code: err.code || null,
        details: err.details || null,
    };
}

/**
 * A progress block for one job: bar, bytes, speed, time left, current file.
 *   const view = progressView('Download progress for Qwen3-VL 32B');
 *   parent.append(view.node); view.update(job);
 */
export function progressView(label) {
    const bar = el('div', { class: 'progress-bar' });
    const track = el('div', {
        class: 'progress-track', role: 'progressbar', 'aria-label': label,
        'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': '0',
    }, bar);
    const main = el('span', { class: 'job-progress-main' });
    const rate = el('span', { class: 'job-progress-rate' });
    const file = el('div', { class: 'job-progress-file' });
    const node = el('div', { class: 'job-progress' }, track, el('div', { class: 'job-progress-stats' }, main, rate), file);

    function update(job) {
        if (!job) return;
        const pct = jobPercent(job);
        const verifying = job.state === 'verifying';
        const shown = verifying ? 100 : pct;
        bar.style.width = `${shown}%`;
        track.classList.toggle('is-verifying', verifying);
        track.setAttribute('aria-valuenow', String(shown));
        let text;
        let right = '';
        if (verifying) {
            text = 'Verifying the files…';
        } else if (job.state === 'queued') {
            const n = Number(job.queue_position);
            text = Number.isInteger(n) && n > 0 ? `Waiting to start (number ${n} in line)…` : 'Waiting to start…';
        } else {
            text = `${fmtBytes(job.bytes_done)} of ${fmtBytes(job.bytes_total)} (${pct}%)`;
            const parts = [];
            if (job.speed_bps > 0) parts.push(`${fmtBytes(job.speed_bps)}/s`);
            if (Number.isFinite(job.eta_seconds) && job.eta_seconds !== null) parts.push(`about ${fmtDuration(job.eta_seconds)} left`);
            right = parts.join(' · ');
        }
        track.setAttribute('aria-valuetext', verifying || job.state === 'queued' ? text : `${pct}%, ${text}${right ? `, ${right}` : ''}`);
        main.textContent = text;
        rate.textContent = right;
        const files = Array.isArray(job.files) ? job.files : [];
        const index = files.findIndex((f) => f.path === job.current_file);
        file.textContent = job.current_file
            ? `File${files.length > 1 && index >= 0 ? ` ${index + 1} of ${files.length}` : ''}: ${job.current_file}`
            : '';
    }
    return { node, update };
}

/** The compact indicator in the page header. Hidden unless a download or install is running. */
export function mountIndicator(host) {
    if (!host) return;
    const text = el('span', { class: 'job-indicator-text' });
    const button = el('button', { type: 'button', class: 'job-indicator', hidden: true },
        el('span', { class: 'spinner spinner-small', 'aria-hidden': 'true' }), text);
    let current = null;
    button.addEventListener('click', () => {
        if (current) gotoTab({ engine_install: 'engine', detector_download: 'setup' }[current.kind] || 'models');
    });
    replace(host, button);
    const render = () => {
        const active = jobs.filter(isActive);
        current = active.find((j) => j.state !== 'queued') || active[0] || null;
        button.hidden = !current;
        if (!current) return;
        const install = current.kind === 'engine_install';
        const detector = current.kind === 'detector_download';
        // Short enough for the narrow header (190 px); the tooltip says the rest.
        const what = install ? 'Engine install' : (detector ? 'Label detector' : `Downloading ${current.label || 'a model'}`);
        const state = current.state === 'verifying' ? 'verifying' : (current.state === 'queued' ? 'waiting' : `${jobPercent(current)}%`);
        const waiting = active.filter((j) => j.kind === 'model_download' && j.state === 'queued' && j !== current).length;
        text.textContent = `${what} · ${state}${waiting ? ` · ${waiting} more queued` : ''}`;
        button.title = install ? 'Installing the engine. Show it on the Engine tab' : (detector ? 'Downloading the label detector. Show it on the Setup page' : 'Show the download on the Models tab');
    };
    on('jobs-changed', render);
    render();
}
