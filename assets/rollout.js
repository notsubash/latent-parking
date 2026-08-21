export function mountRollout(root, spec) {
  const horizon = spec.horizon ?? 8;
  const action = spec.action ?? 1;
  const bias = spec.bias ?? 0.85;

  root.classList.add("widget", "rollout-demo");
  root.innerHTML = "";

  const prompt = document.createElement("p");
  prompt.className = "prompt";
  prompt.textContent = spec.prompt;

  const canvas = document.createElement("canvas");
  canvas.className = "cem-canvas";
  canvas.width = 640;
  canvas.height = 200;

  const toolbar = document.createElement("div");
  toolbar.className = "cem-toolbar";
  const stepBtn = document.createElement("button");
  stepBtn.type = "button";
  stepBtn.textContent = "One open-loop step";
  const resetBtn = document.createElement("button");
  resetBtn.type = "button";
  resetBtn.textContent = "Reset";
  toolbar.append(stepBtn, resetBtn);

  const stats = document.createElement("p");
  stats.className = "feedback";

  root.append(prompt, canvas, toolbar, stats);

  const state = { t: 0, trueX: [0], identX: [0], modelX: [0] };

  function draw() {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const padL = 40;
    const padR = 16;
    const padT = 18;
    const padB = 32;
    const xmax = horizon * action;
    const tToPx = (t) => padL + (t / horizon) * (w - padL - padR);
    const yToPx = (x) => {
      const top = xmax * 1.05;
      return padT + (1 - x / top) * (h - padT - padB);
    };

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#fffcf6";
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = "#d6d3d1";
    ctx.beginPath();
    ctx.moveTo(padL, h - padB);
    ctx.lineTo(w - padR, h - padB);
    ctx.stroke();

    const series = [
      { pts: state.trueX, color: "#1c1917", width: 1.8 },
      { pts: state.identX, color: "#a8a29e", width: 1.4 },
      { pts: state.modelX, color: "#9f1239", width: 1.6 },
    ];
    series.forEach((s) => {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width;
      ctx.beginPath();
      s.pts.forEach((x, i) => {
        const px = tToPx(i);
        const py = yToPx(x);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
      const last = s.pts.length - 1;
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.arc(tToPx(last), yToPx(s.pts[last]), 3.5, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.fillStyle = "#57534e";
    ctx.font = "12px 'Segoe UI', sans-serif";
    ctx.fillText("step", w - padR - 28, h - 10);
    ctx.fillText("x", 8, padT + 8);
    ctx.fillText("true", tToPx(Math.min(state.t, horizon)) + 8, yToPx(state.trueX.at(-1)) - 6);
  }

  function renderStats() {
    if (state.t === 0) {
      stats.className = "feedback";
      stats.textContent =
        "Action is +1 every step. Grey = identity (s′ = s). Crimson = f that underestimates by 15%. Black = truth.";
      return;
    }
    const identErr = Math.abs(state.identX.at(-1) - state.trueX.at(-1));
    const modelErr = Math.abs(state.modelX.at(-1) - state.trueX.at(-1));
    stats.className = "feedback ok";
    stats.textContent =
      `h=${state.t}: identity error=${identErr.toFixed(2)}  biased-f error=${modelErr.toFixed(2)}  (grows with horizon)`;
  }

  function stepOnce() {
    if (state.t >= horizon) return;
    state.trueX.push(state.trueX.at(-1) + action);
    state.identX.push(state.identX.at(-1));
    state.modelX.push(state.modelX.at(-1) + bias * action);
    state.t += 1;
    draw();
    renderStats();
  }

  function resetDemo() {
    state.t = 0;
    state.trueX = [0];
    state.identX = [0];
    state.modelX = [0];
    draw();
    renderStats();
  }

  stepBtn.addEventListener("click", stepOnce);
  resetBtn.addEventListener("click", resetDemo);
  resetDemo();
}
