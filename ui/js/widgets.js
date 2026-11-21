// Reusable interactive widgets (keyboard-accessible tab strips, toggles).

import { el, infoTip } from './dom.js';

let seq = 0;

/**
 * An accessible tab strip.
 *   const t = tabStrip({label: 'Input type', tabs: [{id, label, panel}], initial: 'url', onChange(id)});
 *   container.append(t.list, t.panels);
 * `className` styles the buttons ('tab' for the Step 4 strip, 'result-tab' for results).
 */
export function tabStrip({ label, tabs, initial = null, onChange = null, className = 'tab', listClass = 'tabs', panelClass = 'tab-content' }) {
    const uid = `ts${seq += 1}`;
    const list = el('div', { class: listClass, role: 'tablist', 'aria-label': label });
    const panels = el('div', { class: `${listClass}-panels` });
    const buttons = new Map();
    let current = null;

    for (const t of tabs) {
        const btnId = `${uid}-tab-${t.id}`;
        const panelId = `${uid}-panel-${t.id}`;
        const btn = el('button', {
            type: 'button', role: 'tab', id: btnId, class: className,
            'aria-controls': panelId, 'aria-selected': 'false', tabindex: '-1', dataset: { tab: t.id },
            onclick: () => select(t.id, { focus: false }),
        }, t.label);
        const panel = el('div', { class: panelClass, role: 'tabpanel', id: panelId, 'aria-labelledby': btnId, tabindex: '0' });
        panel.hidden = true;
        if (t.panel) panel.appendChild(t.panel);
        buttons.set(t.id, { btn, panel });
        list.appendChild(btn);
        panels.appendChild(panel);
    }

    list.addEventListener('keydown', (ev) => {
        const ids = [...buttons.keys()];
        const i = ids.indexOf(current);
        let next = null;
        if (ev.key === 'ArrowRight') next = ids[(i + 1) % ids.length];
        else if (ev.key === 'ArrowLeft') next = ids[(i - 1 + ids.length) % ids.length];
        else if (ev.key === 'Home') next = ids[0];
        else if (ev.key === 'End') next = ids[ids.length - 1];
        if (next) {
            ev.preventDefault();
            select(next, { focus: true });
        }
    });

    function select(id, { focus = false, silent = false } = {}) {
        if (!buttons.has(id)) return;
        current = id;
        for (const [tid, { btn, panel }] of buttons) {
            const on = tid === id;
            btn.setAttribute('aria-selected', String(on));
            btn.classList.toggle('active', on);
            btn.tabIndex = on ? 0 : -1;
            panel.hidden = !on;
            panel.classList.toggle('active', on);
        }
        if (focus) buttons.get(id).btn.focus();
        if (onChange && !silent) onChange(id);
    }

    select(initial && buttons.has(initial) ? initial : tabs[0].id, { silent: true });
    return {
        list,
        panels,
        select,
        current: () => current,
        panel: (id) => (buttons.get(id) || {}).panel,
    };
}

/** A labeled checkbox styled like the VoucherVisionGO mode toggles. */
export function modeToggle({ id, label, hint = null, tip = null, checked = false, disabled = false, onchange = null }) {
    const input = el('input', { type: 'checkbox', id, checked, disabled, onchange });
    const hintId = hint ? `${id}-hint` : null;
    if (hintId) input.setAttribute('aria-describedby', hintId);
    return {
        input,
        node: el('div', { class: ['mode-toggle', disabled ? 'is-disabled' : ''] },
            input,
            el('label', { for: id }, label),
            hint ? el('span', { class: 'mode-hint', id: hintId }, hint) : null,
            tip ? infoTip(tip) : null),
    };
}
