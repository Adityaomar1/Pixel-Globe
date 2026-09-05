// Deep Space Animated Canvas Background (Starfield + Cosmic Dust)
(function() {
    const canvas = document.getElementById('starfield-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
        initStars();
    });

    const NUM_STARS = 280;
    const NUM_NEBULA_PARTICLES = 40;
    let stars = [];
    let nebulae = [];
    let speedMultiplier = 1.0;

    class Star {
        constructor() {
            this.reset();
        }
        reset() {
            this.x = (Math.random() - 0.5) * width * 1.5;
            this.y = (Math.random() - 0.5) * height * 1.5;
            this.z = Math.random() * width;
            this.size = Math.random() * 1.8 + 0.5;
            this.twinkle = Math.random() * Math.PI;
            this.twinkleSpeed = 0.02 + Math.random() * 0.03;
            // Color variations: ice blue, warm amber, pulsar cyan, pure white
            const colors = ['#ffffff', '#00f0ff', '#a855f7', '#ffed4a', '#60a5fa'];
            this.color = colors[Math.floor(Math.random() * colors.length)];
        }
        update(speed) {
            this.z -= speed * speedMultiplier;
            this.twinkle += this.twinkleSpeed;
            if (this.z <= 1) {
                this.reset();
                this.z = width;
            }
        }
        draw() {
            const k = 250 / this.z;
            const px = this.x * k + width / 2;
            const py = this.y * k + height / 2;

            if (px >= 0 && px <= width && py >= 0 && py <= height) {
                const alpha = Math.min(1, (1 - this.z / width) * (0.6 + 0.4 * Math.sin(this.twinkle)));
                const sz = Math.max(0.8, this.size * k * 0.8);
                ctx.beginPath();
                ctx.arc(px, py, sz, 0, Math.PI * 2);
                ctx.fillStyle = this.color;
                ctx.globalAlpha = alpha;
                ctx.shadowBlur = sz > 1.8 ? 8 : 0;
                ctx.shadowColor = this.color;
                ctx.fill();
            }
        }
    }

    class NebulaDust {
        constructor() {
            this.reset();
        }
        reset() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.radius = 80 + Math.random() * 160;
            this.vx = (Math.random() - 0.5) * 0.2;
            this.vy = (Math.random() - 0.5) * 0.2;
            const hues = [260, 190, 320, 220]; // Purples, Cyans, Magentas, Blues
            this.hue = hues[Math.floor(Math.random() * hues.length)];
            this.alpha = 0.03 + Math.random() * 0.04;
        }
        update() {
            this.x += this.vx;
            this.y += this.vy;
            if (this.x < -200) this.x = width + 200;
            if (this.x > width + 200) this.x = -200;
            if (this.y < -200) this.y = height + 200;
            if (this.y > height + 200) this.y = -200;
        }
        draw() {
            const grad = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.radius);
            grad.addColorStop(0, `hsla(${this.hue}, 85%, 60%, ${this.alpha})`);
            grad.addColorStop(0.5, `hsla(${this.hue + 20}, 75%, 40%, ${this.alpha * 0.5})`);
            grad.addColorStop(1, 'transparent');

            ctx.globalAlpha = 1;
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function initStars() {
        stars = [];
        nebulae = [];
        for (let i = 0; i < NUM_STARS; i++) stars.push(new Star());
        for (let i = 0; i < NUM_NEBULA_PARTICLES; i++) nebulae.push(new NebulaDust());
    }

    initStars();

    let lastTime = 0;
    function animate(time) {
        ctx.fillStyle = '#050711';
        ctx.globalAlpha = 0.4;
        ctx.fillRect(0, 0, width, height);

        // Draw nebulae
        for (let n of nebulae) {
            n.update();
            n.draw();
        }

        // Draw stars
        for (let s of stars) {
            s.update(1.2);
            s.draw();
        }

        ctx.globalAlpha = 1.0;
        ctx.shadowBlur = 0;
        requestAnimationFrame(animate);
    }

    requestAnimationFrame(animate);

    // Warp speed trigger for hyperspace processing
    window.setHyperspace = function(active) {
        speedMultiplier = active ? 6.0 : 1.0;
    };
})();
