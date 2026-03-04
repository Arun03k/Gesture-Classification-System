const MOVE_STEP = 50;      // Pixels per step
const ZOOM_STEP = 0.2;      // Scaling per step
const ROTATE_STEP = 90;      // Degrees per step

function resetTransform(img) {
  img.dataset.tx = "0";
  img.dataset.ty = "0";
  img.dataset.scale = "1";
  img.dataset.rotation = "0";
  applyTransform(img);
}

function applyTransform(img) {
  const tx = parseFloat(img.dataset.tx || "0");
  const ty = parseFloat(img.dataset.ty || "0");
  const scale = parseFloat(img.dataset.scale || "1");
  const rotation = parseFloat(img.dataset.rotation || "0");

  img.style.transform =
    `translate(${tx}px, ${ty}px) scale(${scale}) rotate(${rotation}deg)`;
}

function getCurrentImage() {
  const slide = Reveal.getCurrentSlide();
  if (!slide) {
    return null;
  }
  return slide.querySelector("img.slide-image");
}

function handleImageCommand(cmd) {
  const img = getCurrentImage();
  if (!img) {
    console.warn("[slideshow] No slide-image found on current slide");
    return;
  }

  const tx = parseFloat(img.dataset.tx || "0");
  const ty = parseFloat(img.dataset.ty || "0");
  const scale = parseFloat(img.dataset.scale || "1");
  const rotation = parseFloat(img.dataset.rotation || "0");

  switch (cmd) {
    case "move_left":
      img.dataset.tx = String(tx - MOVE_STEP);
      break;
    case "move_right":
      img.dataset.tx = String(tx + MOVE_STEP);
      break;
    case "move_up":
      img.dataset.ty = String(ty - MOVE_STEP);
      break;
    case "move_down":
      img.dataset.ty = String(ty + MOVE_STEP);
      break;
    case "rotate_counter_clock":
      img.dataset.rotation = String(rotation - ROTATE_STEP);
      break;
    case "rotate":
      img.dataset.rotation = String(rotation + ROTATE_STEP);
      break;
    case "zoom_in":
      img.dataset.scale = String(scale + ZOOM_STEP);
      break;
    case "zoom_out":
      img.dataset.scale = String(Math.max(0.1, scale - ZOOM_STEP));
      break;
    case "reset":
      resetTransform(img);
      return;
    default:
      console.warn("[slideshow] Unknown image command:", cmd);
      return;
  }

  applyTransform(img);
}

function handleCommand(cmd) {
  switch (cmd) {
    // Navigation
    case "swipe_right":
      Reveal.right();
      break;
    case "swipe_left":
      Reveal.left();
      break;
    case "swipe_up":
      Reveal.up();
      break;
    case "swipe_down":
      Reveal.down();
      break;
    case "first":
      Reveal.slide(0, 0);
      break;
    // Image manipulation
    case "move_left":
    case "move_right":
    case "move_up":
    case "move_down":
    case "rotate":
    case "rotate_counter_clock":
    case "zoom_in":
    case "zoom_out":
    case "reset":
      handleImageCommand(cmd);
      break;

    default:
      console.warn("[slideshow] Unknown command:", cmd);
  }
}

// Helper: HTML for a single slide (image + caption)
function buildSlideContent(section, row, index) {
  const imgPath = row.image;
  const title = row.title;
  const subtitle = row.subtitle;

  if (imgPath) {
    const img = document.createElement("img");
    img.src = imgPath;
    img.className = "slide-image";
    img.alt = title || `Image ${index}`;
    resetTransform(img);
    section.appendChild(img);
  }

  if (title || subtitle) {
    const caption = document.createElement("p");
    caption.className = "caption";

    let text = "";
    if (title) {
      text += title;
    }
    if (subtitle) {
      if (text) {
        text += " – ";
      }
      text += subtitle;
    }

    caption.textContent = text;
    section.appendChild(caption);
  }
}

async function loadSlidesFromCsv() {
  const container = document.getElementById("slides");
  if (!container) {
    console.error("[slideshow] No #slides container found");
    return;
  }

  try {
    const response = await fetch("static/slides.csv");
    if (!response.ok) {
      console.warn("[slideshow] slides.csv not found, showing placeholder slide");
      createFallbackSlide(container, "No slides.csv found");
      return;
    }

    const text = await response.text();
    const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l && !l.startsWith("#"));
    if (lines.length === 0) {
      console.warn("[slideshow] slides.csv is empty, showing placeholder slide");
      createFallbackSlide(container, "slides.csv is empty");
      return;
    }

    const header = lines[0].split(",").map(h => h.trim());
    const imageIndex = header.indexOf("image");
    const titleIndex = header.indexOf("title");
    const subtitleIndex = header.indexOf("subtitle");

    if (imageIndex === -1) {
      console.warn("[slideshow] Column 'image' missing in slides.csv");
      createFallbackSlide(container, "Column 'image' is missing in slides.csv");
      return;
    }

    // Parse CSV rows into objects
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const row = lines[i];
      if (!row) {
        continue;
      }
      const cols = row.split(",").map(c => c.trim());
      const image = cols[imageIndex] || "";
      const title = (titleIndex >= 0 && cols[titleIndex]) ? cols[titleIndex] : "";
      const subtitle = (subtitleIndex >= 0 && cols[subtitleIndex]) ? cols[subtitleIndex] : "";

      rows.push({ image, title, subtitle });
    }

    if (rows.length === 0) {
      createFallbackSlide(container, "No data rows in slides.csv");
      return;
    }

    // GROUPING: consecutive rows with the same `image` → vertical stack
    let i = 0;
    while (i < rows.length) {
      const current = rows[i];
      const group = [current];
      let j = i + 1;
      while (j < rows.length && rows[j].image === current.image) {
        group.push(rows[j]);
        j++;
      }

      if (group.length === 1) {
        // Only one slide with this image → normal horizontal section
        const section = document.createElement("section");
        buildSlideContent(section, current, i);
        container.appendChild(section);
      } else {
        // Multiple slides with the same image → vertical stack
        const parent = document.createElement("section");
        group.forEach((row, idxInGroup) => {
          const sub = document.createElement("section");
          buildSlideContent(sub, row, i + idxInGroup);
          parent.appendChild(sub);
        });
        container.appendChild(parent);
      }

      i = j;
    }
  } catch (err) {
    console.error("[slideshow] Error while loading slides.csv:", err);
    createFallbackSlide(container, "Error while loading slides.csv");
  }
}

function createFallbackSlide(container, message) {
  const section = document.createElement("section");
  const h2 = document.createElement("h2");
  h2.textContent = "Reveal.js Slideshow";
  const p = document.createElement("p");
  p.innerHTML = message + "<br><br>" +
    "Create a file <code>static/slides.csv</code> with columns " +
    "<code>image,title,subtitle</code>.";
  section.appendChild(h2);
  section.appendChild(p);
  container.appendChild(section);
}

function setupWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const url = `${protocol}://${window.location.host}/ws`;

  let ws = new WebSocket(url);

  ws.onopen = () => {
    console.log("[slideshow] WebSocket connected:", url);
  };

  ws.onmessage = (event) => {
    const cmd = String(event.data || "").trim();
    if (cmd) {
      console.log("[slideshow] Command received:", cmd);
      handleCommand(cmd);
    }
  };

  ws.onclose = () => {
    console.warn("[slideshow] WebSocket closed, trying to reconnect in 2s ...");
    setTimeout(setupWebSocket, 2000);
  };

  ws.onerror = () => {
    ws.close();
  };
}

(async function main() {
  await loadSlidesFromCsv();

  // Initialize Reveal after slides are generated
  Reveal.initialize({
    hash: true,
    loop: false,
    plugins: [RevealMarkdown, RevealHighlight, RevealNotes, RevealZoom]
  });

  setupWebSocket();
})();
