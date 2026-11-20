// A small "stored" (uncompressed) ZIP writer. It replaces the JSZip CDN script
// the VoucherVisionGO page loaded on demand. JSON and Markdown results are
// small, so compression is not worth a dependency.

const CRC_TABLE = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n += 1) {
        let c = n;
        for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        t[n] = c >>> 0;
    }
    return t;
})();

function crc32(bytes) {
    let c = 0xffffffff;
    for (let i = 0; i < bytes.length; i += 1) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
}

function dosDateTime(d) {
    const time = (d.getHours() << 11) | (d.getMinutes() << 5) | Math.floor(d.getSeconds() / 2);
    const date = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
    return { time, date };
}

/** Make file names unique and safe inside the archive. */
function uniqueName(name, used) {
    let base = String(name || 'file').replace(/[\\/:*?"<>|\u0000-\u001f]+/g, '_').replace(/^\.+/, '_') || 'file';
    if (!used.has(base)) {
        used.add(base);
        return base;
    }
    const dot = base.lastIndexOf('.');
    const stem = dot > 0 ? base.slice(0, dot) : base;
    const ext = dot > 0 ? base.slice(dot) : '';
    let i = 2;
    while (used.has(`${stem}_${i}${ext}`)) i += 1;
    base = `${stem}_${i}${ext}`;
    used.add(base);
    return base;
}

/** files: [{name, text}] -> Blob (application/zip). */
export function makeZip(files) {
    const enc = new TextEncoder();
    const now = dosDateTime(new Date());
    const used = new Set();
    const chunks = [];
    const central = [];
    let offset = 0;

    for (const f of files) {
        const nameBytes = enc.encode(uniqueName(f.name, used));
        const data = f.bytes instanceof Uint8Array ? f.bytes : enc.encode(String(f.text ?? ''));
        const crc = crc32(data);

        const local = new DataView(new ArrayBuffer(30));
        local.setUint32(0, 0x04034b50, true);
        local.setUint16(4, 20, true);
        local.setUint16(6, 0x0800, true); // UTF-8 names
        local.setUint16(8, 0, true); // stored
        local.setUint16(10, now.time, true);
        local.setUint16(12, now.date, true);
        local.setUint32(14, crc, true);
        local.setUint32(18, data.length, true);
        local.setUint32(22, data.length, true);
        local.setUint16(26, nameBytes.length, true);
        local.setUint16(28, 0, true);
        chunks.push(new Uint8Array(local.buffer), nameBytes, data);

        const cen = new DataView(new ArrayBuffer(46));
        cen.setUint32(0, 0x02014b50, true);
        cen.setUint16(4, 20, true);
        cen.setUint16(6, 20, true);
        cen.setUint16(8, 0x0800, true);
        cen.setUint16(10, 0, true);
        cen.setUint16(12, now.time, true);
        cen.setUint16(14, now.date, true);
        cen.setUint32(16, crc, true);
        cen.setUint32(20, data.length, true);
        cen.setUint32(24, data.length, true);
        cen.setUint16(28, nameBytes.length, true);
        cen.setUint32(42, offset, true);
        central.push(new Uint8Array(cen.buffer), nameBytes);

        offset += 30 + nameBytes.length + data.length;
    }

    const centralSize = central.reduce((n, c) => n + c.length, 0);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true);
    end.setUint16(8, files.length, true);
    end.setUint16(10, files.length, true);
    end.setUint32(12, centralSize, true);
    end.setUint32(16, offset, true);
    return new Blob([...chunks, ...central, new Uint8Array(end.buffer)], { type: 'application/zip' });
}
