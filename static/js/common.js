(function () {
    function createParticles(containerId, particleCount) {
        const container = document.getElementById(containerId);
        if (!container || container.dataset.particlesReady === '1') {
            return;
        }

        container.dataset.particlesReady = '1';
        const count = Number.isFinite(particleCount) ? particleCount : 15;

        for (let index = 0; index < count; index++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 15 + 's';
            particle.style.animationDuration = 15 + Math.random() * 10 + 's';
            container.appendChild(particle);
        }
    }

    window.initializeSharedPageEffects = function (options) {
        const config = options || {};
        createParticles(config.containerId || 'particles', config.particleCount || 15);
        if (window.applySiteTheme) {
            window.applySiteTheme();
        }
    };
})();
