// Keys tab: one card per credential. Key values are write-only: they go from
// the input straight into one request and are never kept in a variable that
// outlives it, in web storage, or in the DOM after saving.

import { api, inlineError, toast, toastError } from '../api.js';
import { el, icon, replace, stateBox, badge, externalLink, confirmDialog, fmtDateTime } from '../dom.js';
import { emit, getAppInfo } from '../state.js';

const GROUPS = [
    ['model_provider', 'Model providers'],
    ['enrichment', 'Data enrichment'],
    ['downloads', 'Downloads'],
];
const TEST_KIND = { ok: 'good', invalid_key: 'bad', not_configured: 'warn', network_error: 'warn', error: 'bad' };
const TEST_LABEL = { ok: 'Key works', invalid_key: 'Key rejected', not_configured: 'No key saved', network_error: 'Could not connect', error: 'Test failed' };

export async function mount(panel) {
    const info = getAppInfo() || {};
    const privateFile = (info.paths && info.paths.private_file) || 'PRIVATE.yaml';
    const body = el('div');
    const cards = new Map();

    function keyCard(initial) {
        const card = el('section', { class: 'card key-card', id: `key-card-${initial.id}`, tabindex: '-1', 'aria-label': initial.label });
        let provider = initial;
        let busy = null;

        function render() {
            const p = provider;
            const inputs = [];
            const fieldNodes = (p.fields || []).map((f) => {
                const id = `key-${p.id}-${f.name}`;
                const input = el('input', {
                    type: 'password', id, name: f.name, placeholder: f.placeholder || '',
                    autocomplete: 'off', spellcheck: 'false', autocapitalize: 'off', disabled: Boolean(busy),
                });
                inputs.push(input);
                const toggle = el('button', { type: 'button', class: 'icon-button', 'aria-label': 'Show what you typed', 'aria-pressed': 'false', title: 'Show or hide what you typed' }, icon('eye', { size: 16 }));
                toggle.addEventListener('click', () => {
                    const show = input.type === 'password';
                    input.type = show ? 'text' : 'password';
                    toggle.setAttribute('aria-pressed', String(show));
                    toggle.setAttribute('aria-label', show ? 'Hide what you typed' : 'Show what you typed');
                    replace(toggle, icon(show ? 'eyeOff' : 'eye', { size: 16 }));
                });
                input.addEventListener('keydown', (ev) => {
                    if (ev.key === 'Enter') save();
                });
                return el('div', { class: 'key-field' },
                    el('label', { for: id }, f.label || f.name,
                        f.saved ? badge('Saved', 'good') : badge('Not saved', 'neutral'),
                        f.saved && f.masked ? el('span', { class: 'key-mask' }, f.masked) : null),
                    el('div', { class: 'key-input-row' }, input, toggle));
            });
            const anySaved = (p.fields || []).some((f) => f.saved);

            async function save() {
                const fields = {};
                for (const input of inputs) {
                    if (input.value.trim()) fields[input.name] = input.value.trim();
                    input.value = '';
                    input.type = 'password';
                }
                if (!Object.keys(fields).length) {
                    toast('Paste a key first.', 'warning');
                    return;
                }
                await act('save', () => api.put(`/v1/keys/${encodeURIComponent(p.id)}`, { fields }), 'Key saved.');
            }

            async function remove() {
                const yes = await confirmDialog({ title: `Remove the ${p.label} key?`, message: `The saved key is deleted from ${privateFile}.`, confirmLabel: 'Remove', danger: true });
                if (yes) await act('remove', () => api.del(`/v1/keys/${encodeURIComponent(p.id)}`), 'Key removed.');
            }

            async function test() {
                busy = 'test';
                render();
                try {
                    const result = await api.post(`/v1/keys/${encodeURIComponent(p.id)}/test`, {}, { timeoutMs: 30000 });
                    provider = { ...provider, last_test: result };
                } catch (err) {
                    toastError(err, `${p.label} key test`);
                } finally {
                    busy = null;
                    render();
                }
            }

            const t = p.last_test;
            replace(card,
                el('div', { class: 'card-header' }, el('h3', null, p.label),
                    p.configured ? badge('Key saved', 'good') : badge('No key saved', 'neutral'),
                    p.optional ? badge('Optional', 'neutral') : null,
                    el('span', { class: 'spacer' }),
                    p.sends_images
                        ? el('span', { class: 'sends-note is-remote' }, icon('cloud', { size: 15 }), 'Images are sent to this service')
                        : el('span', { class: 'sends-note is-local' }, icon('shield', { size: 15 }), 'Never receives your images')),
                p.purpose ? el('p', { class: 'key-purpose' }, p.purpose) : null,
                el('div', { class: 'key-links' },
                    p.key_url ? externalLink('Get a key', p.key_url) : null,
                    p.docs_url ? externalLink('Documentation', p.docs_url) : null),
                fieldNodes,
                el('div', { class: 'key-actions' },
                    el('button', { type: 'button', class: 'button', disabled: Boolean(busy), onclick: save }, busy === 'save' ? 'Saving…' : 'Save'),
                    el('button', { type: 'button', class: 'button-secondary', disabled: Boolean(busy) || !p.configured, onclick: test }, busy === 'test' ? 'Testing…' : 'Test'),
                    el('button', { type: 'button', class: 'button-secondary button-danger', disabled: Boolean(busy) || !anySaved, onclick: remove }, icon('trash', { size: 14 }), 'Remove')),
                t ? el('div', { class: 'key-test', role: 'status' },
                    badge(TEST_LABEL[t.status] || (t.ok ? 'Key works' : 'Test failed'), TEST_KIND[t.status] || (t.ok ? 'good' : 'bad')),
                    el('span', null, t.message || ''),
                    t.tested_at ? el('span', { class: 'key-test-time' }, fmtDateTime(t.tested_at)) : null) : null);
        }

        async function act(kind, call, doneMessage) {
            busy = kind;
            render();
            try {
                const updated = await call();
                if (updated && updated.id) provider = updated;
                toast(doneMessage, 'success');
                emit('providers-changed');
                emit('models-changed');
            } catch (err) {
                toastError(err, provider.label);
            } finally {
                busy = null;
                render();
            }
        }

        render();
        return card;
    }

    async function load() {
        panel.dataset.state = 'loading';
        replace(body, stateBox('loading', 'Loading keys…'));
        cards.clear();
        try {
            const data = await api.get('/v1/providers');
            const providers = (data && data.providers) || [];
            if (!providers.length) {
                replace(body, stateBox('empty', 'No services are listed'));
            } else {
                const known = new Set(GROUPS.map(([role]) => role));
                const groups = [...GROUPS, ['', 'Other']];
                replace(body, groups.map(([role, title]) => {
                    const list = providers.filter((p) => (role ? p.role === role : !known.has(p.role)));
                    if (!list.length) return null;
                    return el('section', { 'aria-label': title }, el('h2', { class: 'section-title' }, title),
                        el('div', { class: 'card-grid' }, list.map((p) => {
                            const card = keyCard(p);
                            cards.set(p.id, card);
                            return card;
                        })));
                }));
            }
            panel.dataset.state = 'ready';
        } catch (err) {
            inlineError(body, err, { title: 'Keys could not be loaded', retry: load });
            panel.dataset.state = 'error';
        }
    }

    replace(panel,
        el('div', { class: 'keys-privacy' }, icon('shield', { size: 28 }),
            el('div', null,
                el('p', null, el('strong', null, 'Keys are stored only on this computer'), ', in ', el('code', null, privateFile), '.'),
                el('p', null, 'A saved key is never shown again. To change one, paste a new key and save.'))),
        body);
    await load();

    return {
        show(target) {
            const card = target ? cards.get(target) : null;
            if (!card) return;
            for (const c of cards.values()) c.classList.remove('is-highlighted');
            card.classList.add('is-highlighted');
            card.scrollIntoView({ block: 'center' });
            card.focus({ preventScroll: true });
        },
    };
}
