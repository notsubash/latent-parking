export function mountTwins(root, spec) {
  const dt = spec.dt ?? 0.2;
  const speed = spec.speed ?? 4;
  const horizon = spec.horizon ?? 8;

  root.classList.add("widget", "twins-demo");
  root.innerHTML = "";

  const prompt = document.createElement("p");
  prompt.className = "prompt";
  prompt.textContent = spec.prompt;

  const legend = document.createElement("div");
  legend.className = "legend-row";
  legend.innerHTML =
    "<span class=\"fwd\">same pose, +v</span><span class=\"rev\">same pose, −v</span>";

  const canvas = document.createElement("canvas");
  canvas.className = "cem-canvas";
  canvas.width = 640;
  canvas.height = 160;

  const toolbar = document.createElement("div");
  toolbar.className = "cem-toolbar";
  const stepBtn = document.createElement("button");
  stepBtn.type = "button";
  stepBtn.textContent = "One 0.2 s step";
  const resetBtn = document.createElement("button");
  resetBtn.type = "button";
  resetBtn.textContent = "Reset";
  toolbar.append(stepBtn, resetBtn);

  const stats = document.createElement("p");
  stats.className = "feedback";

  root.append(prompt, legend, canvas, toolbar, stats);

  const state = { t: 0, fwd: [0], rev: [0] };

  function draw() {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const pad = 28;
    const xmax = speed * dt * horizon;
    const xToPx = (x) => pad + ((x + xmax) / (2 * xmax)) * (w - 2 * pad);
    const y = h / 2;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#fffcf6";
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = "#d6d3d1";
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(w - pad, y);
    ctx.stroke();

    const series = [
      { pts: state.fwd, color: "#14532d" },
      { pts: state.rev, color: "#9f1239" },
    ];
    series.forEach((s) => {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      s.pts.forEach((x, i) => {
        const px = xToPx(x);
        if (i === 0) ctx.moveTo(px, y);
        else ctx.lineTo(px, y);
      });
      ctx.stroke();
      s.pts.forEach((x, i) => {
        ctx.fillStyle = s.color;
        ctx.beginPath();
        ctx.arc(xToPx(x), y, i === s.pts.length - 1 ? 6 : 3, 0, Math.PI * 2);
        ctx.fill();
      });
    });

    ctx.fillStyle = "#57534e";
    ctx.font = "12px 'Segoe UI', sans-serif";
    ctx.fillText("x (m)", w - pad - 28, h - 10);
  }

  function renderStats() {
    if (state.t === 0) {
      stats.className = "feedback";
      stats.textContent =
        "Both cars sit at x=0, heading +x. Throttle and steering are zero. Only the hidden speed differs.";
      return;
    }
    const gap = Math.abs(state.fwd.at(-1) - state.rev.at(-1));
    stats.className = "feedback ok";
    stats.textContent =
      `t=${(state.t * dt).toFixed(1)}s  gap=${gap.toFixed(1)} m. A pose-only f saw one dot at t=0.`;
  }

  function stepOnce() {
    if (state.t >= horizon) return;
    state.fwd.push(state.fwd.at(-1) + speed * dt);
    state.rev.push(state.rev.at(-1) - speed * dt);
    state.t += 1;
    draw();
    renderStats();
  }

  function resetDemo() {
    state.t = 0;
    state.fwd = [0];
    state.rev = [0];
    draw();
    renderStats();
  }

  stepBtn.addEventListener("click", stepOnce);
  resetBtn.addEventListener("click", resetDemo);
  resetDemo();
}
