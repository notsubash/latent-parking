function shuffle(items) {
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

export function mountQuiz(root, questions) {
  root.innerHTML = "";
  questions.forEach((question, index) => {
    const block = document.createElement("div");
    block.className = "quiz";
    const prompt = document.createElement("p");
    prompt.className = "prompt";
    prompt.textContent = `${index + 1}. ${question.prompt}`;
    const choices = document.createElement("div");
    choices.className = "choices";
    const feedback = document.createElement("p");
    feedback.className = "feedback";

    shuffle(question.choices).forEach((choice) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = choice;
      button.addEventListener("click", () => {
        const ok = choice === question.answer;
        Array.from(choices.querySelectorAll("button")).forEach((node) => {
          node.disabled = true;
          if (node.textContent === question.answer) node.classList.add("correct");
        });
        if (!ok) button.classList.add("wrong");
        feedback.className = `feedback ${ok ? "ok" : "bad"}`;
        feedback.textContent = ok ? question.explain : `Not that. ${question.explain}`;
      });
      choices.appendChild(button);
    });

    block.append(prompt, choices, feedback);
    root.appendChild(block);
  });
}

export function mountNumeric(root, spec) {
  root.classList.add("widget");
  root.innerHTML = "";
  const prompt = document.createElement("p");
  prompt.className = "prompt";
  prompt.textContent = spec.prompt;
  const row = document.createElement("div");
  const input = document.createElement("input");
  input.type = "number";
  input.step = "any";
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = "Check";
  const feedback = document.createElement("p");
  feedback.className = "feedback";
  button.addEventListener("click", () => {
    const value = Number(input.value);
    const ok = Number.isFinite(value) && Math.abs(value - spec.answer) <= (spec.tolerance ?? 1e-6);
    feedback.className = `feedback ${ok ? "ok" : "bad"}`;
    feedback.textContent = ok ? spec.explain : spec.hint;
  });
  row.append(input, button);
  root.append(prompt, row, feedback);
}