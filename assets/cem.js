function sampleNormal(mean, std) {
  const u = 1 - Math.random();
  const v = Math.random();
  const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  return mean + std * z;
}

function clip(x, lo, hi) {
  return Math.min(hi, Math.max(lo, x));
}

export function mountCemDemo(root, spec) {
  const target = spec.target ?? 2;
  const lo = spec.lo ?? -0.5;
  const hi = spec.hi ?? 4.5;
  const nSamples = spec.nSamples ?? 24;
  const nElites = spec.nElites ?? 6;
  const initStd = spec.initStd ?? 1.1;
  const cost = spec.cost ?? ((x) => (x - target) ** 2);

  root.classList.add("widget", "cem-demo");
  root.innerHTML = "";

  const prompt = document.createElement("p");
  prompt.className = "prompt";
  prompt.textContent = spec.prompt;

  const canvas = document.createElement("canvas");
  canvas.className = "cem-canvas";
  canvas.width = 640;
  canvas.height = 220;

  const toolbar = document.createElement("div");
  toolbar.className = "cem-toolbar";
  const iterate = document.createElement("button");
  iterate.type = "button";
  iterate.textContent = "One CEM iteration";
  const reset = document.createElement("button");
  reset.type = "button";
  reset.textContent = "Reset";
  toolbar.append(iterate, reset);

  const stats = document.createElement("p");
  stats.className = "feedback";

  root.append(prompt, canvas, toolbar, stats);

  const state = {
    mean: spec.initMean ?? 0,
    std: initStd,
    iter: 0,
    samples: [],
    elites: [],
  };

  function draw() {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const padL = 36;
    const padR = 16;
    const padT = 16;
    const padB = 36;
    const xToPx = (x) => padL + ((x - lo) / (hi - lo)) * (w - padL - padR);
    const yToPx = (y) => {
      const ymax = cost(lo - 0.2);
      return padT + (1 - y / ymax) * (h - padT - padB);
    };

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#fffcf6";
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = "#1c1917";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i <= 80; i += 1) {
      const x = lo + (i / 80) * (hi - lo);
      const px = xToPx(x);
      const py = yToPx(cost(x));
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    ctx.strokeStyle = "#a8a29e";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(xToPx(target), padT);
    ctx.lineTo(xToPx(target), h - padB);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#57534e";
    ctx.font = "12px 'Segoe UI', sans-serif";
    ctx.fillText("min", xToPx(target) + 6, padT + 12);

    const axisY = h - padB;
    ctx.strokeStyle = "#d6d3d1";
    ctx.beginPath();
    ctx.moveTo(padL, axisY);
    ctx.lineTo(w - padR, axisY);
    ctx.stroke();

    state.samples.forEach((x) => {
      const elite = state.elites.includes(x);
      ctx.fillStyle = elite ? "#9f1239" : "#a8a29e";
      ctx.beginPath();
      ctx.arc(xToPx(x), yToPx(cost(x)), elite ? 5 : 3.5, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.strokeStyle = "#14532d";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(xToPx(state.mean), padT);
    ctx.lineTo(xToPx(state.mean), h - padB);
    ctx.stroke();

    ctx.fillStyle = "#57534e";
    ctx.fillText("x", w - padR - 10, h - 12);
    ctx.fillText("cost", 4, padT + 10);
  }

  function renderStats() {
    if (state.iter === 0 && state.samples.length === 0) {
      stats.className = "feedback";
      stats.textContent = `mean=${state.mean.toFixed(2)}  std=${state.std.toFixed(2)}  (sample around a bad guess)`;
      return;
    }
    stats.className = "feedback ok";
    stats.textContent =
      `iter ${state.iter}: mean=${state.mean.toFixed(2)}  std=${state.std.toFixed(2)}  ` +
      `elites=${nElites}/${nSamples} (crimson). Green line is the new mean.`;
  }

  function iterateOnce() {
    const samples = [];
    for (let i = 0; i < nSamples; i += 1) {
      samples.push(clip(sampleNormal(state.mean, state.std), lo, hi));
    }
    samples.sort((a, b) => cost(a) - cost(b));
    const elites = samples.slice(0, nElites);
    state.samples = samples;
    state.elites = elites;
    state.mean = elites.reduce((s, x) => s + x, 0) / elites.length;
    const variance = elites.reduce((s, x) => s + (x - state.mean) ** 2, 0) / elites.length;
    state.std = Math.max(Math.sqrt(variance), 0.04);
    state.iter += 1;
    draw();
    renderStats();
  }

  function resetDemo() {
    state.mean = spec.initMean ?? 0;
    state.std = initStd;
    state.iter = 0;
    state.samples = [];
    state.elites = [];
    draw();
    renderStats();
  }

  iterate.addEventListener("click", iterateOnce);
  reset.addEventListener("click", resetDemo);
  resetDemo();
}
