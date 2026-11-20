// Results map: vendored Leaflet (ui/vendor/leaflet) with OpenStreetMap raster
// tiles. The tiles are the only thing this UI downloads from the internet;
// without a connection the map area shows a plain message instead.

import { el } from '../dom.js';

const BASE = '/ui/vendor/leaflet/';
const TILE_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';

let leafletPromise = null;

/** Load the vendored Leaflet stylesheet and script once; resolves with window.L. */
function loadLeaflet() {
    if (!leafletPromise) {
        leafletPromise = new Promise((resolve, reject) => {
            if (window.L && window.L.map) {
                resolve(window.L);
                return;
            }
            document.head.appendChild(el('link', { rel: 'stylesheet', href: `${BASE}leaflet.css` }));
            const script = el('script', { src: `${BASE}leaflet.js` });
            script.addEventListener('load', () => (window.L && window.L.map ? resolve(window.L) : reject(new Error('Leaflet did not load'))));
            script.addEventListener('error', () => reject(new Error('Leaflet did not load')));
            document.head.appendChild(script);
        }).catch((err) => {
            leafletPromise = null;
            throw err;
        });
    }
    return leafletPromise;
}

/**
 * createMap(container) -> {setPoint(lat, lon, label), clearPoint(), resize()}
 * The map is built on the first setPoint, so nothing is requested until a
 * result has coordinates.
 */
export function createMap(container) {
    const mapNode = el('div', { class: 'map-viewer', role: 'application', 'aria-label': 'Map of the specimen coordinates' });
    const offline = el('div', { class: 'map-offline', role: 'status', hidden: true }, 'Map tiles unavailable offline');
    const holder = el('div', { class: 'map-frame', hidden: true }, mapNode, offline);
    container.appendChild(holder);

    let map = null;
    let marker = null;
    let loadedTiles = 0;
    let failedTiles = 0;

    function updateOffline() {
        offline.hidden = !(failedTiles > 0 && loadedTiles === 0);
    }

    async function ensureMap() {
        if (map) return map;
        const L = await loadLeaflet();
        if (map) return map;
        map = L.map(mapNode, { center: [0, 0], zoom: 1, worldCopyJump: true });
        map.attributionControl.setPrefix(false);
        const tiles = L.tileLayer(TILE_URL, {
            maxZoom: 19,
            attribution: '© OpenStreetMap contributors',
            referrerPolicy: 'no-referrer',
        });
        tiles.on('tileload', () => {
            loadedTiles += 1;
            updateOffline();
        });
        tiles.on('tileerror', () => {
            failedTiles += 1;
            updateOffline();
        });
        tiles.addTo(map);
        window.addEventListener('online', () => {
            failedTiles = 0;
            updateOffline();
            tiles.redraw();
        });
        return map;
    }

    return {
        async setPoint(lat, lon, label = null) {
            holder.hidden = false;
            let m;
            try {
                m = await ensureMap();
            } catch (_) {
                offline.textContent = 'The map could not be loaded';
                offline.hidden = false;
                return;
            }
            const L = window.L;
            if (marker) marker.remove();
            marker = L.marker([lat, lon], {
                icon: L.icon({
                    iconUrl: `${BASE}images/marker-icon.png`,
                    iconRetinaUrl: `${BASE}images/marker-icon-2x.png`,
                    shadowUrl: `${BASE}images/marker-shadow.png`,
                    iconSize: [25, 41],
                    iconAnchor: [12, 41],
                    popupAnchor: [1, -34],
                    shadowSize: [41, 41],
                }),
                keyboard: false,
            }).addTo(m);
            if (label) marker.bindPopup(el('span', null, label));
            m.invalidateSize();
            m.setView([lat, lon], 9);
        },
        clearPoint() {
            if (marker) marker.remove();
            marker = null;
            holder.hidden = true;
        },
        resize() {
            if (map) map.invalidateSize();
        },
    };
}
