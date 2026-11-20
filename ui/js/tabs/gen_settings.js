// Model settings inside the OCR and Parsing cards of the VVL tab (GET/PUT /v1/generation).
//
// One row of chips under each card's model list, showing only what the chosen
// models support:
//   Reasoning      on/off chip; for models with strength levels (Muse) a menu
//                  next to it picks the strength while reasoning is on
//   Long output    on/off chip (standard / long output token limit)
//   High detail    on/off chip, OCR only (image tokens; restarts the engine)
//   Context        chip with a menu: a size per model, or Automatic
//   { }            the literal values the current choices send
// The explanations are hover and focus hints (data-tip), not text on the page.

import { api, toastError } from '../api.js';
import { el, icon, replace } from '../dom.js';

const STEPS = ['ocr', 'parse'];
const STEP_NAME = { ocr: 'OCR', parse: 'Parsing' };
const LEVEL_LABEL = { low: 'Low', medium: 'Medium', high: 'High', xhigh: 'Extra high' };
const LEVEL_NOTE = { low: 'fastest', high: 'model default', xhigh: 'slowest' };
const fmt = (n) => Number(n).toLocaleString('en-US');
const fmtTokens = (n) => `${fmt(n)} tokens`;
const fmtK = (n) => (n >= 1024 && n % 1024 === 0 ? `${n / 1024}K` : fmt(n));
const listNames = (names) => (names.length <= 1 ? names.join('')
    : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`);
const notFor = (names) => (names.length ? ` Does not apply to ${listNames(names)}.` : '');

/**
 * mountGenSettings({ ocr: cardNode, parse: cardNode }) -> { refresh() }
 * The row is appended to the end of each card.
 */
export function mountGenSettings(cards) {
    const state = { data: null, saving: false, pop: null }; // pop: { step, name } of the open menu
    const rows = {};
    for (const step of STEPS) {
        rows[step] = el('div', { class: 'gen-row', role: 'group', 'aria-label': `${STEP_NAME[step]} model settings`, hidden: true });
        cards[step].appendChild(rows[step]);
    }

    // A click outside an open menu, or Escape, closes it.
    document.addEventListener('mousedown', (ev) => {
        if (state.pop && !ev.target.closest('.gen-pop-root')) closePop(false);
    });
    document.addEventListener('keydown', (ev) => {
        if (ev.key === 'Escape' && state.pop) closePop(true);
    });

    function closePop(returnFocus) {
        const was = state.pop;
        state.pop = null;
        render();
        if (returnFocus && was) {
            const trigger = rows[was.step].querySelector(`[data-fk="${was.step}-${was.name}"]`);
            if (trigger) trigger.focus();
        }
    }

    function togglePop(step, name) {
        state.pop = state.pop && state.pop.step === step && state.pop.name === name ? null : { step, name };
        render();
    }

    const isOpen = (step, name) => !!(state.pop && state.pop.step === step && state.pop.name === name);

    // Controls stay enabled while saving (disabling them would drop keyboard
    // focus); a second change waits until the first is saved.
    async function save(body) {
        if (state.saving) return;
        state.saving = true;
        for (const step of STEPS) rows[step].setAttribute('aria-busy', 'true');
        try {
            state.data = await api.put('/v1/generation', body);
        } catch (err) {
            toastError(err, 'The setting was not saved');
        } finally {
            state.saving = false;
            for (const step of STEPS) rows[step].removeAttribute('aria-busy');
            render();
        }
    }

    // ------------------------------------------------------------ hints
    function hints(step, block, ctx) {
        const models = block.models;
        const lacking = (key) => models.filter((m) => !m.controls[key]).map((m) => m.label);
        const keepsReasoning = models.filter((m) => m.controls.reasoning && m.controls.reasoning.can_turn_off === false)
            .map((m) => m.label);
        let reasoning = 'The model thinks before it answers. Slower, and the thinking counts toward the output length.';
        if (keepsReasoning.length) reasoning += ` ${listNames(keepsReasoning)} always reasons a little; with this off it uses its lowest strength.`;
        reasoning += notFor(lacking('reasoning'));

        const strength = `How long the model thinks before it answers. Higher is slower.${notFor(lacking('strength'))}`;

        // Output limits can differ per model (a model that always reasons gets more room).
        const limits = models.map((m) => ({ label: m.label, ...((m.controls.output_length || {}).max_tokens || {}) }));
        const same = limits.every((l) => l.standard === limits[0].standard && l.long === limits[0].long);
        let output = 'Most tokens one reply may use, thinking included.';
        if (same && limits.length) output += ` Standard: up to ${fmt(limits[0].standard)}. Long: up to ${fmt(limits[0].long)}.`;
        else output += ` Standard / long: ${limits.map((l) => `${l.label} ${fmt(l.standard)} / ${fmt(l.long)}`).join('; ')}.`;

        let detail = '';
        const withDetail = models.find((m) => m.controls.image_detail);
        if (withDetail) {
            const tokens = withDetail.controls.image_detail.tokens || {};
            const std = (tokens.standard || {}).image_max_tokens;
            const high = (tokens.high || {}).image_max_tokens;
            detail = (std && high ? `High sends each image at up to ${fmt(high)} visual tokens instead of ${fmt(std)}: finer print, slower.`
                : 'High sends each image at a higher resolution: finer print, slower.')
                + ' Changing it restarts the engine at the next image.' + notFor(lacking('image_detail'));
        }

        const context = 'How much text the model holds at once: prompt, image and reply. A larger window uses more graphics memory; changing it restarts the engine. '
            + ctx.map((c) => (ctx.length > 1 ? `${c.label}: ` : '') + contextNote(c)).join(' ');
        return { reasoning, strength, output, detail, context };
    }

    function contextNote(entry) {
        const max = Number(entry.max) || null;
        if (entry.value === 'auto' && entry.mode === 'auto_max') {
            return `Automatic: ${entry.effective ? fmtTokens(entry.effective) : 'as large as fits'}, the most that fits in free graphics memory now`
                + (max ? ` (model maximum ${fmt(max)})` : '') + ', because reasoning is on.';
        }
        if (entry.value === 'auto') return `Automatic: ${fmtTokens(entry.effective || entry.needed || entry.min || 8192)}, enough for these settings.`;
        return `Set to ${fmtTokens(entry.value)}. Range ${fmt(entry.min || 8192)} to ${max ? fmt(max) : 'the model maximum'}.`;
    }

    // ------------------------------------------------------------ controls
    function toggleChip(step, key, on, label, tip, onValue, offValue, align) {
        return el('button', {
            type: 'button', class: 'gen-chip', 'aria-pressed': on ? 'true' : 'false',
            'data-tip': tip, 'data-tip-align': align || null, 'aria-description': tip, 'data-fk': `${step}-${key}`,
            onclick: () => save({ [step]: { [key]: on ? offValue : onValue } }),
        }, on ? icon('check', { size: 12 }) : null, label);
    }

    function strengthMenu(step, block, tip) {
        const v = block.values || {};
        const levels = (block.models.find((m) => m.controls.strength) || {}).controls.strength.options;
        const current = levels.includes(v.strength) ? v.strength : levels[levels.length - 1];
        const open = isOpen(step, 'strength');
        const menu = open ? el('div', { class: 'gen-pop gen-menu', role: 'menu', 'aria-label': 'Reasoning strength' },
            levels.map((level) => el('button', {
                type: 'button', role: 'menuitemradio', class: ['gen-menu-item', level === current ? 'is-on' : ''],
                'aria-checked': level === current ? 'true' : 'false', 'data-fk': `${step}-level-${level}`,
                onclick: () => { state.pop = null; if (level !== current) save({ [step]: { strength: level } }); else render(); },
            }, LEVEL_LABEL[level] || level, el('span', { class: 'gen-menu-sub' }, LEVEL_NOTE[level] || '')))) : null;
        return el('span', { class: 'gen-pop-root' },
            el('button', {
                type: 'button', class: 'gen-chip is-on', 'aria-haspopup': 'menu', 'aria-expanded': open ? 'true' : 'false',
                'aria-label': `Reasoning strength: ${LEVEL_LABEL[current] || current}`, 'data-tip': tip, 'aria-description': tip,
                'data-fk': `${step}-strength`, onclick: () => togglePop(step, 'strength'),
            }, LEVEL_LABEL[current] || current, icon('chevronDown', { size: 12 })),
            menu);
    }

    function contextControl(step, entry, multi) {
        const auto = entry.value === 'auto';
        const max = Number(entry.max) || null;
        const id = `gen-ctx-${step}-${entry.model_id.replace(/[^a-z0-9]/gi, '_')}`;
        const input = el('input', {
            type: 'number', id, class: 'gen-ctx-input', min: String(entry.min || 8192), max: max ? String(max) : null,
            step: '1024', inputmode: 'numeric', disabled: auto, 'data-fk': `${step}-ctxn-${entry.model_id}`,
            value: String(auto ? (entry.effective || entry.min || 8192) : entry.value),
            'aria-label': `Context window for ${entry.label}, in tokens`,
        });
        const commit = () => {
            let n = Math.round(Number(input.value) / 1024) * 1024;
            if (!Number.isFinite(n)) return;
            n = Math.max(Number(entry.min) || 8192, max ? Math.min(max, n) : n);
            input.value = String(n);
            if (n !== entry.value) save({ context: { [entry.model_id]: n } });
        };
        input.addEventListener('change', commit);
        input.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') { ev.preventDefault(); commit(); } });
        const autoBox = el('input', {
            type: 'checkbox', id: `${id}-auto`, checked: auto, 'data-fk': `${step}-ctxa-${entry.model_id}`,
            onchange: () => save({ context: { [entry.model_id]: autoBox.checked ? 'auto' : (entry.effective || entry.min || 8192) } }),
        });
        return el('div', null,
            multi ? el('span', { class: 'gen-ctx-model' }, entry.label) : null,
            el('div', { class: 'gen-ctx-line' }, input,
                el('label', { class: 'check-row gen-ctx-auto', for: `${id}-auto` }, autoBox, 'Automatic')),
            el('p', { class: 'gen-pop-note' }, contextNote(entry)));
    }

    function contextChip(step, ctx, tip) {
        const open = isOpen(step, 'context');
        const allAuto = ctx.every((c) => c.value === 'auto');
        const text = allAuto ? 'Auto' : ctx.length === 1 ? fmtK(ctx[0].value) : 'Custom';
        return el('span', { class: 'gen-pop-root' },
            el('button', {
                type: 'button', class: ['gen-chip', allAuto ? '' : 'is-on'], 'aria-haspopup': 'dialog',
                'aria-expanded': open ? 'true' : 'false', 'data-tip': tip, 'aria-description': tip,
                'data-fk': `${step}-context`, onclick: () => togglePop(step, 'context'),
            }, `Context: ${text}`, icon('chevronDown', { size: 12 })),
            open ? el('div', { class: 'gen-pop', role: 'dialog', 'aria-label': 'Context window' },
                el('div', { class: 'gen-pop-title' }, 'Context window'),
                el('div', { class: 'gen-ctx-list' }, ctx.map((c) => contextControl(step, c, ctx.length > 1)))) : null);
    }

    function rawButton(step, block, ctx) {
        const open = isOpen(step, 'raw');
        let pop = null;
        if (open) {
            const sent = {};
            for (const m of block.models) sent[m.label] = { ...m.sent };
            for (const c of ctx) (sent[c.label] || (sent[c.label] = {})).context_window = c.effective || null;
            pop = el('div', { class: 'gen-pop is-end', role: 'dialog', 'aria-label': 'Values sent to the model' },
                el('div', { class: 'gen-pop-title' }, 'Values sent to the model'),
                el('pre', { class: 'logs logs-small gen-raw-pre', tabindex: '0' }, JSON.stringify(sent, null, 2)));
        }
        return el('span', { class: 'gen-pop-root gen-end' },
            el('button', {
                type: 'button', class: 'gen-icon-btn', 'aria-haspopup': 'dialog', 'aria-expanded': open ? 'true' : 'false',
                'aria-label': 'Values sent to the model', 'data-tip': 'Values sent to the model', 'data-tip-align': 'end',
                'data-fk': `${step}-raw`, onclick: () => togglePop(step, 'raw'),
            }, '{ }'),
            pop);
    }

    // ------------------------------------------------------------ render
    function renderStep(step) {
        const row = rows[step];
        const data = state.data;
        const block = data && data[step];
        if (!block || !block.models.length) {
            row.hidden = true;
            replace(row);
            return;
        }
        row.hidden = false;
        const v = block.values || {};
        const app = block.applicable || {};
        const ctx = ((data.context && data.context.models) || []).filter((c) => (c.steps || []).includes(step));
        const tip = hints(step, block, ctx);
        const items = [];
        if (app.reasoning) {
            const chip = toggleChip(step, 'reasoning', v.reasoning === 'on', 'Reasoning', tip.reasoning, 'on', 'off', 'start');
            items.push(app.strength && v.reasoning === 'on'
                ? el('span', { class: 'gen-pair' }, chip, strengthMenu(step, block, tip.strength))
                : chip);
        }
        if (app.output_length !== false) {
            items.push(toggleChip(step, 'output_length', v.output_length === 'long', 'Long output', tip.output, 'long', 'standard',
                app.reasoning ? null : 'start'));
        }
        if (step === 'ocr' && app.image_detail) {
            items.push(toggleChip(step, 'image_detail', v.image_detail === 'high', 'High detail', tip.detail, 'high', 'standard'));
        }
        if (ctx.length) items.push(contextChip(step, ctx, tip.context));
        items.push(rawButton(step, block, ctx));
        replace(row, items);
    }

    function render() {
        // Keep keyboard focus on the same control across the redraw.
        const active = document.activeElement;
        const fk = active && active.dataset ? active.dataset.fk : null;
        for (const step of STEPS) renderStep(step);
        if (fk) {
            const again = document.querySelector(`.gen-row [data-fk="${CSS.escape(fk)}"]`);
            if (again && again !== document.activeElement) again.focus({ preventScroll: true });
        }
    }

    async function refresh() {
        try {
            state.data = await api.get('/v1/generation', { timeoutMs: 30000 });
        } catch (_) {
            state.data = null; // the cards still work without these settings
        }
        render();
    }

    return { refresh };
}
