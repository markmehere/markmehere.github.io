let session;
let meta;
let fakes = null;

async function loadFakes() {
  const response = await fetch("fakes.json");
  if (!response.ok) throw new Error("Failed to load fakes.json");
  fakes = await response.json();
}

async function loadModel() {
  session = await ort.InferenceSession.create("austen.onnx", {
    executionProviders: ["webgpu"],
  });
  meta = await fetch("meta.json").then(res => res.json());
}

// Create an int64 token sequence
function makeInputTensor(blockSize) {
  const data = new BigInt64Array(blockSize);
  data.fill(0n); // or your BOS token
  return new ort.Tensor("int64", data, [1, blockSize]);
}

async function forward(xTensor) {
  const outputs = await session.run({ x: xTensor });
  return outputs[Object.keys(outputs)[0]]; // logits
}

async function fakeGenerate(
  steps = 500
) {
  if (!fakes || !fakes.length) {
    return 'A person who can write a long letter with ease, cannot write ill.\n\nAlas, at the moment, I cannot write either.\n\n404 - file not found.';
  }

  let text = fakes[Math.floor(Math.random() * fakes.length)];

  const dangerZone = steps - 20;
  if (text.length > dangerZone) {
    const indexOr99 = (s) => s === -1 ? 99 : s;
    const endZone = text.slice(dangerZone);
    let cut = Math.min(
      indexOr99(endZone.indexOf(' ')),
      indexOr99(endZone.indexOf('.')),
      indexOr99(endZone.indexOf(',')),
      indexOr99(endZone.indexOf('\n'))
    ) + dangerZone;
    if (cut < text.length) {
      const otext = text;
      text = text.slice(0, cut);
      if (otext[cut] === '.' || otext[cut] === ' ' || otext[cut] === ',') text += '.';
    }
  }

  return text;
}

async function generate(
  blockSize = 256,
  steps = 500,
  temperature = 1.0,
  topK = 20
) {
  let result = '', quickExit = false;
  let x = makeInputTensor(blockSize);

  for (let i = 0; i < steps && !quickExit; i++) {
    const logits = await forward(x);

    const vocabSize = logits.dims[2];
    const lastLogits = logits.data.slice(-vocabSize);

    const nextId = sampleToken(lastLogits, temperature, topK);

    if (i > steps - 20 && (meta.itos[nextId] === '.' || meta.itos[nextId] === ' ' || meta.itos[nextId] === ',')) {
      quickExit = true;
      result += '.';
    }
    else if (i > steps - 20 && meta.itos[nextId] === '\n') {
      quickExit = true;
      result += '\n';
    }
    else {
      result += meta.itos[nextId];
    }

    // shift window + append next token ID
    const old = x.data;
    const shifted = old.slice(1); // drop first token

    const updated = new BigInt64Array(blockSize);
    updated.set(shifted);
    updated[blockSize - 1] = BigInt(nextId);

    x = new ort.Tensor("int64", updated, [1, blockSize]);
  }

  return result;
}

function sampleToken(logits, temperature, topK) {
  const scaled = logits.map(v => v / temperature);

  // top‑k
  const indices = [...scaled.keys()];
  indices.sort((a, b) => scaled[b] - scaled[a]);
  const top = indices.slice(0, topK);

  const probs = top.map(i => Math.exp(scaled[i]));
  const sum = probs.reduce((a, b) => a + b, 0);
  const norm = probs.map(p => p / sum);

  // sample
  let r = Math.random();
  let cum = 0;
  for (let i = 0; i < norm.length; i++) {
    cum += norm[i];
    if (r < cum) return top[i];
  }

  return top[top.length - 1];
}

function fixQuotations(input) {
  const paragraphs = input.split('\n');

  const doubleCapMatch = paragraphs[0].match(/([A-Z]{2,})/);
  if (paragraphs.length > 3 && paragraphs[0].length < 50) {
    paragraphs.shift();
  } 
  else if (doubleCapMatch) {
    const idx = paragraphs[0].search(/[A-Z]{2,}/);
    if (idx !== -1 && idx < 24) {
      const afterDoubleCap = paragraphs[0].slice(idx);
      const wordMatch = afterDoubleCap.match(/\b\w+\b/);
      if (wordMatch) {
        const cutIdx = idx + wordMatch.index + wordMatch[0].length;
        let newPara = paragraphs[0].slice(cutIdx).trimStart();
        if (newPara.length > 0) {
          newPara = newPara[0].toUpperCase() + newPara.slice(1);
          paragraphs[0] = newPara;
        }
      }
    }
  }

  try {
    for (let i = 0; i < paragraphs.length; i++) {
      let text = paragraphs[i];
      text = text.replace(/ ?-- ?/g, '\u2014');

      let quoteCount = 0;
      text = text.replace(/"/g, () => {
        quoteCount++;
        return quoteCount % 2 ? '\u201c' : '\u201d';
      });

      if (i === paragraphs.length - 1) {
        if (quoteCount % 2) text = text.trim() + '\u201d';
        else text = text.trim();
      } else {
        if (quoteCount % 2) {
          const lastCurly = Math.max(text.lastIndexOf('\u201c'), text.lastIndexOf('\u201d'));
          if (lastCurly !== -1) {
            text = text.slice(0, lastCurly) + text.slice(lastCurly + 1);
          }
        }
        text = text.trim();
      }
      paragraphs[i] = text;
    }
  } catch (e) {
    return input;
  }

  return paragraphs.join('\n');
}

async function prepare() {
  await document.fonts.ready;
  document.getElementById("loading").classList.remove("hide");
  let generation;
  try {
    await loadModel();
    generation = await generate();
  }
  catch (e) {
    window.faking = true;
    await loadFakes();
    generation = await fakeGenerate();
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  const result = fixQuotations(generation);

  const rsplit = result.split('\n');
  const credits = window.faking ? [
    "As most mobile phones disable WebGPU, this demo",
    "Was mostly a lie",

    "But it was an entertaining lie! Isn't that the real truth?",
    "No! Please run it on a computer with Chrome",
    
    "It still used the work of",
    "Jane Austen, akelley6 and Josh Nichols",

    "And if it worked it would have used the code of",
    "Andrej Karpathy and Mark Pazolli",

    "Download the source code on GitHub!"
  ] : [
    "This nonsense was written",
    "On your own computer",

    "Using the code of",
    "Andrej Karpathy and Mark Pazolli",
    
    "Which produced a model trained on the works of",
    "Jane Austen",

    "With sound and design elements by",
    "akelley6 and Josh Nichols",

    "Download the source code on GitHub!"
  ];
  window.prose = result;
  window.credits = credits.join('\n').replace('WebGPU', 'Web G.P.U.');
  document.getElementById("letter").innerHTML = (rsplit.map(line => `<p class="prose">${line}</p>`).join('\n') + '\n'
   + credits.map((line, i) => `<p class="${i % 2 ? 'majorCredit' : 'minorCredit'} hidden removed">${line}</p>`).join('\n'))
   .replace("GitHub", "<a href=\"https://github.com/markmehere/nonsense-austen\">GitHub</a>");
  document.getElementById("envelope").classList.remove("loading");
  document.getElementById("loading").classList.add("hide");
  console.log(result);
}
