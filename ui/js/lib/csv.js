// Minimal RFC 4180 CSV reader and writer (replaces the Papa Parse CDN script
// the VoucherVisionGO page loaded at runtime).

/** Parse CSV text into rows of strings. Handles quotes, escaped quotes, CRLF and a BOM. */
export function parseCsvRows(text) {
    const s = String(text).replace(/^﻿/, '');
    const rows = [];
    let row = [];
    let field = '';
    let inQuotes = false;
    for (let i = 0; i < s.length; i += 1) {
        const c = s[i];
        if (inQuotes) {
            if (c === '"') {
                if (s[i + 1] === '"') {
                    field += '"';
                    i += 1;
                } else {
                    inQuotes = false;
                }
            } else {
                field += c;
            }
        } else if (c === '"') {
            inQuotes = true;
        } else if (c === ',') {
            row.push(field);
            field = '';
        } else if (c === '\n' || c === '\r') {
            if (c === '\r' && s[i + 1] === '\n') i += 1;
            row.push(field);
            rows.push(row);
            row = [];
            field = '';
        } else {
            field += c;
        }
    }
    if (field !== '' || row.length) {
        row.push(field);
        rows.push(row);
    }
    return rows.filter((r) => r.some((v) => v.trim() !== ''));
}

/** Parse CSV with a header row into {headers, records: [{header: value}]}. */
export function parseCsv(text) {
    const rows = parseCsvRows(text);
    if (!rows.length) return { headers: [], records: [] };
    const headers = rows[0].map((h) => h.trim());
    const records = rows.slice(1).map((r) => {
        const rec = {};
        headers.forEach((h, i) => {
            rec[h] = r[i] !== undefined ? r[i] : '';
        });
        return rec;
    });
    return { headers, records };
}

/** Quote one CSV cell. Objects are written as JSON. A leading =,+,-,@ is kept as-is (no Excel guard). */
export function csvCell(value) {
    if (value === undefined || value === null) return '';
    const s = typeof value === 'object' ? JSON.stringify(value) : String(value);
    return /[",\r\n]/.test(s) || /^\s|\s$/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function toCsv(headers, rows) {
    const lines = [headers.map(csvCell).join(',')];
    for (const r of rows) lines.push(headers.map((h) => csvCell(r[h])).join(','));
    return `${lines.join('\r\n')}\r\n`;
}
