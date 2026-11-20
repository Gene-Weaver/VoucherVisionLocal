// Runs before first paint (classic script in <head>) so the page never flashes
// the wrong theme. The full theme logic lives in app.js.
(function () {
    var theme = null;
    try {
        var raw = window.localStorage.getItem('vvl.theme');
        theme = raw ? JSON.parse(raw) : null;
    } catch (e) {
        theme = null;
    }
    if (theme !== 'dark' && theme !== 'light') {
        theme = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    document.documentElement.setAttribute('data-theme', theme);
})();
