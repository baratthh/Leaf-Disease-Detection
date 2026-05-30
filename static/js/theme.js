(function () {
    const THEME_ATTR = 'data-theme';
    const STORAGE_KEY = 'fuzzy-theme';

    function getSavedTheme() {
        try {
            return localStorage.getItem(STORAGE_KEY);
        } catch (error) {
            return null;
        }
    }

    function setTheme(mode) {
        const theme = mode === 'dark' ? 'dark' : 'light';
        const isLight = theme === 'light';
        const root = document.documentElement;
        root.setAttribute(THEME_ATTR, theme);
        document.body.classList.remove('light-mode');

        const label = document.getElementById('themeToggleLabel');
        const icon = document.getElementById('themeToggleIcon');
        if (label) {
            label.textContent = isLight ? 'Dark' : 'Light';
        }
        if (icon) {
            icon.innerHTML = isLight
                ? '<circle cx="12" cy="12" r="4.5" stroke="currentColor" stroke-width="1.5"/><path d="M12 2v2M12 20v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M2 12h2M20 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>'
                : '<path d="M21 12.8A8.5 8.5 0 1111.2 3 7 7 0 0021 12.8z" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>';
        }
        try {
            localStorage.setItem(STORAGE_KEY, theme);
        } catch (error) {
            // Ignore storage errors and keep the in-memory theme.
        }
    }

    window.applySiteTheme = function () {
        setTheme(getSavedTheme() || 'dark');
        const toggle = document.getElementById('themeToggleBtn');
        if (toggle && !toggle.dataset.bound) {
            toggle.dataset.bound = '1';
            toggle.addEventListener('click', function () {
                const current = document.documentElement.getAttribute(THEME_ATTR) || 'light';
                setTheme(current === 'light' ? 'dark' : 'light');
            });
        }
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', window.applySiteTheme, { once: true });
    } else {
        window.applySiteTheme();
    }
})();
