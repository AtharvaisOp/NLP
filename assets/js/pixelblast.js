(function (global) {
    const hexToRGB = hex => {
        const c = hex.replace('#', '').padEnd(6, '0');
        const r = parseInt(c.slice(0, 2), 16) / 255;
        const g = parseInt(c.slice(2, 4), 16) / 255;
        const b = parseInt(c.slice(4, 6), 16) / 255;
        return [r, g, b];
    };

    const createTouchTexture = () => {
        const size = 64;
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        const trail = [];
        let last = null;
        const maxAge = 64;
        let radius = 0.1 * size;
        const speed = 1 / maxAge;

        const clear = () => {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        };

        const drawPoint = p => {
            const pos = { x: p.x * size, y: (1 - p.y) * size };
            let intensity = 1;
            const easeOutSine = t => Math.sin((t * Math.PI) / 2);
            const easeOutQuad = t => -t * (t - 2);
            if (p.age < maxAge * 0.3) intensity = easeOutSine(p.age / (maxAge * 0.3));
            else intensity = easeOutQuad(1 - (p.age - maxAge * 0.3) / (maxAge * 0.7)) || 0;
            intensity *= p.force;
            const color = `${((p.vx + 1) / 2) * 255}, ${((p.vy + 1) / 2) * 255}, ${intensity * 255}`;
            const offset = size * 5;
            ctx.shadowOffsetX = offset;
            ctx.shadowOffsetY = offset;
            ctx.shadowBlur = radius;
            ctx.shadowColor = `rgba(${color},${0.22 * intensity})`;
            ctx.beginPath();
            ctx.fillStyle = 'rgba(255,0,0,1)';
            ctx.arc(pos.x - offset, pos.y - offset, radius, 0, Math.PI * 2);
            ctx.fill();
        };

        const addTouch = norm => {
            let force = 0;
            let vx = 0;
            let vy = 0;
            if (last) {
                const dx = norm.x - last.x;
                const dy = norm.y - last.y;
                if (dx === 0 && dy === 0) return;
                const dd = dx * dx + dy * dy;
                const d = Math.sqrt(dd);
                vx = dx / (d || 1);
                vy = dy / (d || 1);
                force = Math.min(dd * 10000, 1);
            }
            last = { x: norm.x, y: norm.y };
            trail.push({ x: norm.x, y: norm.y, age: 0, force, vx, vy });
        };

        const update = () => {
            clear();
            for (let i = trail.length - 1; i >= 0; i--) {
                const point = trail[i];
                const f = point.force * speed * (1 - point.age / maxAge);
                point.x += point.vx * f;
                point.y += point.vy * f;
                point.age++;
                if (point.age > maxAge) trail.splice(i, 1);
            }
            for (let i = 0; i < trail.length; i++) drawPoint(trail[i]);
        };

        return {
            canvas,
            addTouch,
            update,
            set radiusScale(v) { radius = 0.1 * size * v; }
        };
    };

    const SHAPE_MAP = {
        square: 0,
        circle: 1,
        triangle: 2,
        diamond: 3
    };

    const vertexSource = `
  attribute vec2 position;
  void main() {
      gl_Position = vec4(position, 0.0, 1.0);
  }
  `;

    const fragmentSource = `
  #extension GL_OES_standard_derivatives : enable
  precision highp float;

  uniform vec3  uColor;
  uniform vec2  uResolution;
  uniform float uTime;
  uniform float uPixelSize;
  uniform float uScale;
  uniform float uDensity;
  uniform float uPixelJitter;
  uniform int   uEnableRipples;
  uniform float uRippleSpeed;
  uniform float uRippleThickness;
  uniform float uRippleIntensity;
  uniform float uEdgeFade;

  uniform int   uShapeType;
  const int SHAPE_SQUARE   = 0;
  const int SHAPE_CIRCLE   = 1;
  const int SHAPE_TRIANGLE = 2;
  const int SHAPE_DIAMOND  = 3;

  const int   MAX_CLICKS = 10;
  uniform vec2  uClickPos[10];
  uniform float uClickTimes[10];

  uniform sampler2D uTouchTexture;
  uniform int uLiquidEnabled;
  uniform float uLiquidStrength;
  uniform float uLiquidFreq;

  float Bayer2(vec2 a) {
      a = floor(a);
      return fract(a.x / 2. + a.y * a.y * .75);
  }
  #define Bayer4(a) (Bayer2(.5*(a))*0.25 + Bayer2(a))
  #define Bayer8(a) (Bayer4(.5*(a))*0.25 + Bayer2(a))

  #define FBM_OCTAVES     5
  #define FBM_LACUNARITY  1.25
  #define FBM_GAIN        1.0

  float hash11(float n){ return fract(sin(n)*43758.5453); }

  float vnoise(vec3 p){
      vec3 ip = floor(p);
      vec3 fp = fract(p);
      float n000 = hash11(dot(ip + vec3(0.0,0.0,0.0), vec3(1.0,57.0,113.0)));
      float n100 = hash11(dot(ip + vec3(1.0,0.0,0.0), vec3(1.0,57.0,113.0)));
      float n010 = hash11(dot(ip + vec3(0.0,1.0,0.0), vec3(1.0,57.0,113.0)));
      float n110 = hash11(dot(ip + vec3(1.0,1.0,0.0), vec3(1.0,57.0,113.0)));
      float n001 = hash11(dot(ip + vec3(0.0,0.0,1.0), vec3(1.0,57.0,113.0)));
      float n101 = hash11(dot(ip + vec3(1.0,0.0,1.0), vec3(1.0,57.0,113.0)));
      float n011 = hash11(dot(ip + vec3(0.0,1.0,1.0), vec3(1.0,57.0,113.0)));
      float n111 = hash11(dot(ip + vec3(1.0,1.0,1.0), vec3(1.0,57.0,113.0)));
      vec3 w = fp*fp*fp*(fp*(fp*6.0-15.0)+10.0);
      float x00 = mix(n000, n100, w.x);
      float x10 = mix(n010, n110, w.x);
      float x01 = mix(n001, n101, w.x);
      float x11 = mix(n011, n111, w.x);
      float y0  = mix(x00, x10, w.y);
      float y1  = mix(x01, x11, w.y);
      return mix(y0, y1, w.z) * 2.0 - 1.0;
  }

  float fbm2(vec2 uv, float t){
      vec3 p = vec3(uv * uScale, t);
      float amp = 1.0;
      float freq = 1.0;
      float sum = 1.0;
      for (int i = 0; i < FBM_OCTAVES; ++i){
          sum  += amp * vnoise(p * freq);
          freq *= FBM_LACUNARITY;
          amp  *= FBM_GAIN;
      }
      return sum * 0.5 + 0.5;
  }

  float maskCircle(vec2 p, float cov){
      float r = sqrt(cov) * .25;
      float d = length(p - 0.5) - r;
      float aa = 0.5 * fwidth(d);
      return cov * (1.0 - smoothstep(-aa, aa, d * 2.0));
  }

  float maskTriangle(vec2 p, vec2 id, float cov){
      bool flip = mod(id.x + id.y, 2.0) > 0.5;
      if (flip) p.x = 1.0 - p.x;
      float r = sqrt(cov);
      float d  = p.y - r*(1.0 - p.x);
      float aa = fwidth(d);
      return cov * clamp(0.5 - d/aa, 0.0, 1.0);
  }

  float maskDiamond(vec2 p, float cov){
      float r = sqrt(cov) * 0.564;
      return step(abs(p.x - 0.49) + abs(p.y - 0.49), r);
  }

  void main(){
      vec2 texUv = gl_FragCoord.xy / uResolution;

      if (uLiquidEnabled == 1) {
          vec4 tex = texture2D(uTouchTexture, texUv);
          float vx = tex.r * 2.0 - 1.0;
          float vy = tex.g * 2.0 - 1.0;
          float intensity = tex.b;
          float wave = 0.5 + 0.5 * sin(uTime * uLiquidFreq + intensity * 6.2831853);
          float amt = uLiquidStrength * intensity * wave;
          // Apply distortion
          texUv += vec2(vx, vy) * amt;
      }

      float pixelSize = uPixelSize;
      vec2 fragCoord = texUv * uResolution - uResolution * .5;
      float aspectRatio = uResolution.x / uResolution.y;

      vec2 pixelId = floor(fragCoord / pixelSize);
      vec2 pixelUV = fract(fragCoord / pixelSize);

      float cellPixelSize = 8.0 * pixelSize;
      vec2 cellId = floor(fragCoord / cellPixelSize);
      vec2 cellCoord = cellId * cellPixelSize;
      vec2 uv = cellCoord / uResolution * vec2(aspectRatio, 1.0);

      float base = fbm2(uv, uTime * 0.05);
      base = base * 0.5 - 0.65;

      float feed = base + (uDensity - 0.5) * 0.3;

      float speed     = uRippleSpeed;
      float thickness = uRippleThickness;
      const float dampT     = 1.0;
      const float dampR     = 10.0;

      if (uEnableRipples == 1) {
          for (int i = 0; i < 10; ++i){
              vec2 pos = uClickPos[i];
              if (pos.x < 0.0) continue;
              float cellPixelSize = 8.0 * pixelSize;
              vec2 cuv = (((pos - uResolution * .5 - cellPixelSize * .5) / (uResolution))) * vec2(aspectRatio, 1.0);
              float t = max(uTime - uClickTimes[i], 0.0);
              float r = distance(uv, cuv);
              float waveR = speed * t;
              float ring  = exp(-pow((r - waveR) / thickness, 2.0));
              float atten = exp(-dampT * t) * exp(-dampR * r);
              feed = max(feed, ring * atten * uRippleIntensity);
          }
      }

      float bayer = Bayer8(fragCoord / uPixelSize) - 0.5;
      float bw = step(0.5, feed + bayer);

      float h = fract(sin(dot(floor(fragCoord / uPixelSize), vec2(127.1, 311.7))) * 43758.5453);
      float jitterScale = 1.0 + (h - 0.5) * uPixelJitter;
      float coverage = bw * jitterScale;
      float M;
      if      (uShapeType == SHAPE_CIRCLE)   M = maskCircle (pixelUV, coverage);
      else if (uShapeType == SHAPE_TRIANGLE) M = maskTriangle(pixelUV, pixelId, coverage);
      else if (uShapeType == SHAPE_DIAMOND)  M = maskDiamond(pixelUV, coverage);
      else                                   M = coverage;

      if (uEdgeFade > 0.0) {
          vec2 norm = texUv;
          float edge = min(min(norm.x, norm.y), min(1.0 - norm.x, 1.0 - norm.y));
          float fade = smoothstep(0.0, uEdgeFade, edge);
          M *= fade;
      }

      vec3 color = uColor;
      vec3 srgbColor = mix(
          color * 12.92,
          1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055,
          step(vec3(0.0031308), color)
      );

      gl_FragColor = vec4(srgbColor * M, M);
  }
  `;

    function createShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error(gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    function createProgram(gl, vSource, fSource) {
        const vShader = createShader(gl, gl.VERTEX_SHADER, vSource);
        const fShader = createShader(gl, gl.FRAGMENT_SHADER, fSource);
        const prog = gl.createProgram();
        gl.attachShader(prog, vShader);
        gl.attachShader(prog, fShader);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            console.error(gl.getProgramInfoLog(prog));
            return null;
        }
        return prog;
    }

    class PixelBlast {
        constructor(container, options = {}) {
            this.container = container;
            this.options = {
                variant: 'circle',
                pixelSize: 6,
                color: '#B497CF',
                patternScale: 3,
                patternDensity: 1.2,
                liquid: true,
                liquidStrength: 0.12,
                liquidRadius: 1.2,
                pixelSizeJitter: 0.5,
                enableRipples: true,
                rippleIntensityScale: 1.5,
                rippleThickness: 0.12,
                rippleSpeed: 0.4,
                liquidWobbleSpeed: 5,
                speed: 0.6,
                edgeFade: 0.25,
                ...options
            };
            this.dpr = Math.min(window.devicePixelRatio || 1, 2);

            this.canvas = document.createElement('canvas');
            this.canvas.style.width = '100%';
            this.canvas.style.height = '100%';
            this.canvas.style.display = 'block';
            this.container.appendChild(this.canvas);

            this.gl = this.canvas.getContext('webgl', { alpha: true, antialias: false });
            if (!this.gl) return;

            // Required extension for derivatives (fwidth)
            this.gl.getExtension('OES_standard_derivatives');

            this.program = createProgram(this.gl, vertexSource, fragmentSource);
            this.gl.useProgram(this.program);

            const vertices = new Float32Array([
                -1.0, -1.0,
                3.0, -1.0,
                -1.0, 3.0
            ]);
            this.positionBuffer = this.gl.createBuffer();
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
            this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);

            const posLoc = this.gl.getAttribLocation(this.program, "position");
            this.gl.enableVertexAttribArray(posLoc);
            this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);

            this.uniforms = {
                uResolution: this.gl.getUniformLocation(this.program, "uResolution"),
                uTime: this.gl.getUniformLocation(this.program, "uTime"),
                uColor: this.gl.getUniformLocation(this.program, "uColor"),
                uShapeType: this.gl.getUniformLocation(this.program, "uShapeType"),
                uPixelSize: this.gl.getUniformLocation(this.program, "uPixelSize"),
                uScale: this.gl.getUniformLocation(this.program, "uScale"),
                uDensity: this.gl.getUniformLocation(this.program, "uDensity"),
                uPixelJitter: this.gl.getUniformLocation(this.program, "uPixelJitter"),
                uEnableRipples: this.gl.getUniformLocation(this.program, "uEnableRipples"),
                uRippleSpeed: this.gl.getUniformLocation(this.program, "uRippleSpeed"),
                uRippleThickness: this.gl.getUniformLocation(this.program, "uRippleThickness"),
                uRippleIntensity: this.gl.getUniformLocation(this.program, "uRippleIntensity"),
                uEdgeFade: this.gl.getUniformLocation(this.program, "uEdgeFade"),
                uLiquidEnabled: this.gl.getUniformLocation(this.program, "uLiquidEnabled"),
                uLiquidStrength: this.gl.getUniformLocation(this.program, "uLiquidStrength"),
                uLiquidFreq: this.gl.getUniformLocation(this.program, "uLiquidFreq"),
                uTouchTexture: this.gl.getUniformLocation(this.program, "uTouchTexture")
            };

            this.MAX_CLICKS = 10;
            this.clicks = Array.from({ length: this.MAX_CLICKS }, () => ({ pos: [-1, -1], time: 0 }));
            this.clickIx = 0;

            this.uClickPosLocs = [];
            this.uClickTimesLocs = [];
            for (let i = 0; i < this.MAX_CLICKS; i++) {
                this.uClickPosLocs.push(this.gl.getUniformLocation(this.program, `uClickPos[${i}]`));
                this.uClickTimesLocs.push(this.gl.getUniformLocation(this.program, `uClickTimes[${i}]`));
            }

            if (this.options.liquid) {
                this.touch = createTouchTexture();
                this.touch.radiusScale = this.options.liquidRadius;

                this.touchTexture = this.gl.createTexture();
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.touchTexture);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
                this.gl.uniform1i(this.uniforms.uTouchTexture, 0);
            }

            this.setupUniforms();

            this.resizeObserver = new ResizeObserver(() => this.resize());
            this.resizeObserver.observe(this.container);
            this.resize();

            this.onPointerDown = e => {
                const rect = this.canvas.getBoundingClientRect();
                const scaleX = this.canvas.width / rect.width;
                const scaleY = this.canvas.height / rect.height;
                const fx = (e.clientX - rect.left) * scaleX;
                const fy = (rect.height - (e.clientY - rect.top)) * scaleY;

                this.clicks[this.clickIx].pos = [fx, fy];
                this.clicks[this.clickIx].time = this.currentTime;
                this.clickIx = (this.clickIx + 1) % this.MAX_CLICKS;
            };

            this.onPointerMove = e => {
                if (!this.touch) return;
                const rect = this.canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) / rect.width;
                const y = 1.0 - (e.clientY - rect.top) / rect.height;
                this.touch.addTouch({ x, y });
            };

            this.canvas.addEventListener('pointerdown', this.onPointerDown);
            this.canvas.addEventListener('pointermove', this.onPointerMove);

            this.timeOffset = Math.random() * 1000;
            this.loop = this.loop.bind(this);
            this.rafId = requestAnimationFrame(this.loop);
        }

        setupUniforms() {
            const o = this.options;
            const col = hexToRGB(o.color);
            this.gl.uniform3f(this.uniforms.uColor, col[0], col[1], col[2]);
            this.gl.uniform1i(this.uniforms.uShapeType, SHAPE_MAP[o.variant] || 0);
            this.gl.uniform1f(this.uniforms.uPixelSize, o.pixelSize * this.dpr);
            this.gl.uniform1f(this.uniforms.uScale, o.patternScale);
            this.gl.uniform1f(this.uniforms.uDensity, o.patternDensity);
            this.gl.uniform1f(this.uniforms.uPixelJitter, o.pixelSizeJitter);
            this.gl.uniform1i(this.uniforms.uEnableRipples, o.enableRipples ? 1 : 0);
            this.gl.uniform1f(this.uniforms.uRippleSpeed, o.rippleSpeed);
            this.gl.uniform1f(this.uniforms.uRippleThickness, o.rippleThickness);
            this.gl.uniform1f(this.uniforms.uRippleIntensity, o.rippleIntensityScale);
            this.gl.uniform1f(this.uniforms.uEdgeFade, o.edgeFade);
            this.gl.uniform1i(this.uniforms.uLiquidEnabled, o.liquid ? 1 : 0);
            this.gl.uniform1f(this.uniforms.uLiquidStrength, o.liquidStrength);
            this.gl.uniform1f(this.uniforms.uLiquidFreq, o.liquidWobbleSpeed);
        }

        resize() {
            if (!this.gl) return;
            const rect = this.container.getBoundingClientRect();
            const w = Math.floor(rect.width * this.dpr);
            const h = Math.floor(rect.height * this.dpr);
            this.canvas.width = w;
            this.canvas.height = h;
            this.gl.viewport(0, 0, w, h);
            this.gl.uniform2f(this.uniforms.uResolution, w, h);
        }

        loop(t) {
            this.rafId = requestAnimationFrame(this.loop);
            if (!this.gl) return;

            this.currentTime = this.timeOffset + t * 0.001 * this.options.speed;

            this.gl.uniform1f(this.uniforms.uTime, this.currentTime);

            for (let i = 0; i < this.MAX_CLICKS; i++) {
                this.gl.uniform2f(this.uClickPosLocs[i], this.clicks[i].pos[0], this.clicks[i].pos[1]);
                this.gl.uniform1f(this.uClickTimesLocs[i], this.clicks[i].time);
            }

            if (this.touch) {
                this.touch.update();
                this.gl.activeTexture(this.gl.TEXTURE0);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.touchTexture);
                this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, this.touch.canvas);
            }

            this.gl.clearColor(0, 0, 0, 0);
            this.gl.clear(this.gl.COLOR_BUFFER_BIT);
            this.gl.drawArrays(this.gl.TRIANGLES, 0, 3);
        }
    }

    global.initPixelBlastBackgrounds = () => {
        const heroSections = document.querySelectorAll('.hero, .page-hero');
        if (heroSections.length === 0) return;

        heroSections.forEach(hero => {
            if (hero.querySelector('.pixel-blast-container')) return;

            const container = document.createElement('div');
            container.className = 'pixel-blast-container';
            container.style.position = 'absolute';
            container.style.top = '0';
            container.style.left = '0';
            container.style.width = '100%';
            container.style.height = '100%';
            container.style.overflow = 'hidden';
            container.style.zIndex = '0';
            container.style.pointerEvents = 'auto'; // allow mouse events for ripples/liquid
            hero.insertBefore(container, hero.firstChild);

            // Ensure siblings appear above background
            Array.from(hero.children).forEach(child => {
                if (child !== container) {
                    const style = window.getComputedStyle(child);
                    if (style.position === 'static') child.style.position = 'relative';
                    if (style.zIndex === 'auto' || parseInt(style.zIndex, 10) <= 0) child.style.zIndex = '1';
                }
            });

            new PixelBlast(container, {
                variant: 'circle',
                pixelSize: 6,
                color: '#818cf8', // Indigo theme color
                patternScale: 3,
                patternDensity: 1.2,
                liquid: true,
                liquidStrength: 0.12,
                liquidRadius: 1.2,
                pixelSizeJitter: 0.5,
                enableRipples: true,
                rippleIntensityScale: 1.5,
                rippleThickness: 0.12,
                rippleSpeed: 0.4,
                liquidWobbleSpeed: 5,
                speed: 0.6,
                edgeFade: 0.25,
            });
        });
    };

})(window);
