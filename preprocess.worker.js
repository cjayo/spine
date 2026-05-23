function otsuThreshold(data, width, height) {
  const hist = new Array(256).fill(0);
  const totalPixels = width * height;
  for (let i = 0; i < data.length; i += 4) hist[data[i]]++;
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0, wB = 0, maxVariance = 0, threshold = 0;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = totalPixels - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const variance = wB * wF * (mB - mF) ** 2;
    if (variance > maxVariance) { maxVariance = variance; threshold = t; }
  }
  return threshold;
}

// ── Connected-component filter (the expensive part) ──
function filterComponents(data, width, height, minArea, maxArea) {
  const totalPixels = width * height;
  const labels = new Int32Array(totalPixels).fill(-1);
  const componentSizes = [];
  let labelCount = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (data[idx * 4] === 0 && labels[idx] === -1) {
        // Iterative BFS instead of recursive
        const queue = [idx];
        let size = 0;
        labels[idx] = labelCount;
        let head = 0;

        while (head < queue.length) {
          const cur = queue[head++];
          size++;
          const cx = cur % width, cy = (cur - cx) / width;

          if (cy > 0) {
            const n = cur - width;
            if (data[n * 4] === 0 && labels[n] === -1) { labels[n] = labelCount; queue.push(n); }
          }
          if (cy < height - 1) {
            const n = cur + width;
            if (data[n * 4] === 0 && labels[n] === -1) { labels[n] = labelCount; queue.push(n); }
          }
          if (cx > 0) {
            const n = cur - 1;
            if (data[n * 4] === 0 && labels[n] === -1) { labels[n] = labelCount; queue.push(n); }
          }
          if (cx < width - 1) {
            const n = cur + 1;
            if (data[n * 4] === 0 && labels[n] === -1) { labels[n] = labelCount; queue.push(n); }
          }
        }
        componentSizes.push(size);
        labelCount++;
      }
    }
  }

  const keep = new Set();
  for (let i = 0; i < componentSizes.length; i++) {
    if (componentSizes[i] >= minArea && componentSizes[i] <= maxArea) keep.add(i);
  }

  const out = new Uint8ClampedArray(data.length);
  for (let i = 0; i < totalPixels; i++) {
    const pi = i * 4;
    const val = (data[pi] === 0 && keep.has(labels[i])) ? 0 : 255;
    out[pi] = out[pi + 1] = out[pi + 2] = val;
    out[pi + 3] = 255;
  }
  return new ImageData(out, width, height);
}

// ── Main processing ──
function preprocess(imageData, minArea, maxArea) {
  const { width, height, data } = imageData;
  const len = data.length;

  // 1. Grayscale
  for (let i = 0; i < len; i += 4) {
    const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    data[i] = data[i + 1] = data[i + 2] = gray;
  }

  // 2. Contrast stretch (2nd–98th percentile)
  const grays = [];
  for (let i = 0; i < len; i += 4) grays.push(data[i]);
  grays.sort((a, b) => a - b);
  const lo = grays[Math.floor(grays.length * 0.02)];
  const hi = grays[Math.floor(grays.length * 0.98)];
  const range = Math.max(hi - lo, 1);
  for (let i = 0; i < len; i += 4) {
    const stretched = Math.max(0, Math.min(255, ((data[i] - lo) / range) * 255));
    data[i] = data[i + 1] = data[i + 2] = stretched;
  }

  // 3. Otsu binarize
  const otsu = otsuThreshold(data, width, height);
  for (let i = 0; i < len; i += 4) {
    const val = data[i] > otsu ? 255 : 0;
    data[i] = data[i + 1] = data[i + 2] = val;
  }

  // 4. Invert if mostly black
  let blackCount = 0;
  const totalPixels = len / 4;
  for (let i = 0; i < len; i += 4) if (data[i] === 0) blackCount++;
  if (blackCount / totalPixels > 0.5) {
    for (let i = 0; i < len; i += 4) {
      data[i] = 255 - data[i];
      data[i + 1] = 255 - data[i + 1];
      data[i + 2] = 255 - data[i + 2];
    }
  }

  // 5. Connected-component filter
  return filterComponents(data, width, height, minArea, maxArea);
}

// ── Message handler ──
self.onmessage = function (e) {
  const { id, imageData, minArea, maxArea } = e.data;
  try {
    // imageData.buffer was transferred — we own it now
    const result = preprocess(imageData, minArea, maxArea);
    // Transfer the buffer back so the main thread gets it zero-copy
    self.postMessage({ id, imageData: result }, [result.data.buffer]);
  } catch (err) {
    self.postMessage({ id, error: err.message });
  }
};