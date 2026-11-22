// Independent Web Worker for decoding embedded base64 images off the main thread.
// Used when viewing local JSON files dropped/selected directly in the browser.

function detectMime(bytes) {
    if (bytes.length >= 2 && bytes[0] === 0xff && bytes[1] === 0xd8) return 'image/jpeg';
    if (bytes.length >= 4 && bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) return 'image/png';
    if (bytes.length >= 12 && bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46
        && bytes[8] === 0x57 && bytes[9] === 0x45 && bytes[10] === 0x42 && bytes[11] === 0x50) return 'image/webp';
    if (bytes.length >= 4 && bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x38) return 'image/gif';
    return 'image/jpeg';
}

self.onmessage = (ev) => {
    const { id, base64 } = ev.data || {};
    if (!base64 || typeof base64 !== 'string') {
        self.postMessage({ id, ok: false, error: 'Empty image data' });
        return;
    }
    try {
        let clean = base64.trim();
        if (clean.startsWith('data:')) {
            const comma = clean.indexOf(',');
            if (comma >= 0) clean = clean.slice(comma + 1).trim();
        }
        const bin = atob(clean);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
        const mime = detectMime(bytes);
        self.postMessage({ id, ok: true, buffer: bytes.buffer, mime }, [bytes.buffer]);
    } catch (err) {
        self.postMessage({ id, ok: false, error: err && err.message ? err.message : 'Decode failed' });
    }
};
