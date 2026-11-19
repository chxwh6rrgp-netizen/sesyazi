// Speaker features + clustering. Works in the browser (global SPK) and in Node.
(function (root) {
  const SR = 16000, N = 512, FR = 400, HOP = 160, NM = 40, NC = 20;
  const WIN = 2.0, STEP = 1.0; // analysis window (s) and hop (s)
  const win = new Float32Array(FR); for (let i = 0; i < FR; i++) win[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (FR - 1));
  const mel = f => 2595 * Math.log10(1 + f / 700), imel = m => 700 * (10 ** (m / 2595) - 1);
  const lo = mel(80), hi = mel(7600), pts = []; for (let i = 0; i < NM + 2; i++) pts.push(Math.floor((N + 1) * imel(lo + (hi - lo) * i / (NM + 1)) / SR));
  const fb = []; for (let m = 1; m <= NM; m++) { const a = pts[m - 1], b = pts[m], c = pts[m + 1], w = []; for (let k = a; k < c; k++) w.push([k, k < b ? (k - a) / Math.max(1, b - a) : (c - k) / Math.max(1, c - b)]); fb.push(w); }
  const dct = []; for (let k = 1; k < NC; k++) { const r = new Float32Array(NM); for (let n = 0; n < NM; n++) r[n] = Math.cos(Math.PI * k * (n + .5) / NM); dct.push(r); }
  const rev = new Uint16Array(N); for (let i = 0; i < N; i++) { let r = 0; for (let b = 0, x = i; b < 9; b++) { r = (r << 1) | (x & 1); x >>= 1; } rev[i] = r; }
  const cosT = new Float32Array(N / 2), sinT = new Float32Array(N / 2); for (let i = 0; i < N / 2; i++) { cosT[i] = Math.cos(2 * Math.PI * i / N); sinT[i] = -Math.sin(2 * Math.PI * i / N); }
  const re = new Float32Array(N), im = new Float32Array(N), lm = new Float32Array(NM);
  function fft() {
    for (let i = 0; i < N; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let size = 2; size <= N; size <<= 1) { const half = size >> 1, step = N / size; for (let i = 0; i < N; i += size) { for (let j = 0; j < half; j++) { const k = j * step, a = i + j, b = a + half; const tr = re[b] * cosT[k] - im[b] * sinT[k], ti = re[b] * sinT[k] + im[b] * cosT[k]; re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti; } } }
  }
  // per-frame MFCC + log energy
  function frame(pcm, o, out) {
    let pre = o > 0 ? pcm[o - 1] / 32768 : 0, e = 0;
    for (let i = 0; i < FR; i++) { const v = pcm[o + i] / 32768; re[i] = (v - 0.97 * pre) * win[i]; im[i] = 0; pre = v; e += v * v; }
    for (let i = FR; i < N; i++) { re[i] = 0; im[i] = 0; }
    fft();
    for (let m = 0; m < NM; m++) { let a = 1e-10; const w = fb[m]; for (let q = 0; q < w.length; q++) { const k = w[q][0]; a += (re[k] * re[k] + im[k] * im[k]) * w[q][1]; } lm[m] = Math.log(a); }
    for (let k = 0; k < NC - 1; k++) { let a = 0; const r = dct[k]; for (let n = 0; n < NM; n++) a += r[n] * lm[n]; out[k] = a; }
    return Math.log(e / FR + 1e-10);
  }
  function pitch(pcm, c) { // autocorrelation, 8 kHz decimated 40 ms -> Hz or 0
    const L = 320, s = c - L; if (s < 0 || c + L > pcm.length) return 0; const x = new Float32Array(L); let e = 0;
    for (let i = 0; i < L; i++) { const v = (pcm[s + 2 * i] + pcm[s + 2 * i + 1]) / 65536; x[i] = v; e += v * v; } if (e < 1e-4) return 0;
    let best = 0, bl = 0; for (let lag = 16; lag <= 110; lag++) { let r = 0; for (let i = 0; i + lag < L; i++) r += x[i] * x[i + lag]; r /= e; if (r > best) { best = r; bl = lag; } }
    return best > 0.45 ? 8000 / bl : 0;
  }
  // Analyse [s,e) samples: returns windows [{t, f:number[]}] for windows with enough speech
  function windows(pcm, s, e, speechThr) {
    s = Math.max(0, s | 0); e = Math.min(pcm.length, e | 0);
    const nF = Math.max(0, Math.floor((e - s - FR) / HOP));
    const C = new Array(nF), E = new Float32Array(nF), tmp = new Float32Array(NC - 1);
    for (let f = 0; f < nF; f++) { E[f] = frame(pcm, s + f * HOP, tmp); C[f] = Float32Array.from(tmp); }
    let thr = speechThr;
    if (thr == null) { const so = Array.from(E).sort((a, b) => a - b); const p10 = so[Math.floor(so.length * .1)] ?? -20, p95 = so[Math.floor(so.length * .95)] ?? 0; thr = p10 + 0.45 * (p95 - p10); }
    const out = [], wf = Math.round(WIN * SR / HOP), sf = Math.round(STEP * SR / HOP), D = NC - 1;
    for (let a = 0; a + wf <= nF || (a === 0 && nF > 30); a += sf) {
      const b = Math.min(nF, a + wf); const idx = []; for (let f = a; f < b; f++) if (E[f] > thr) idx.push(f);
      if (idx.length < (b - a) * 0.45 || idx.length < 25) { if (b >= nF) break; continue; }
      const mean = new Float64Array(D), sd = new Float64Array(D);
      for (const f of idx) for (let k = 0; k < D; k++) mean[k] += C[f][k];
      for (let k = 0; k < D; k++) mean[k] /= idx.length;
      for (const f of idx) for (let k = 0; k < D; k++) { const d = C[f][k] - mean[k]; sd[k] += d * d; }
      for (let k = 0; k < D; k++) sd[k] = Math.sqrt(sd[k] / idx.length);
      const f0 = []; for (let i = 0; i < idx.length; i += 2) { const p = pitch(pcm, s + idx[i] * HOP + FR / 2); if (p > 55 && p < 480) f0.push(Math.log(p)); }
      f0.sort((x, y) => x - y);
      const med = f0.length >= 5 ? f0[f0.length >> 1] : NaN, iqr = f0.length >= 5 ? f0[Math.floor(f0.length * .8)] - f0[Math.floor(f0.length * .2)] : NaN;
      out.push({ t: (s + (a + (b - a) / 2) * HOP + FR / 2) / SR, f: [...Array.from(mean, v => +v.toFixed(3)), ...Array.from(sd, v => +v.toFixed(3)), isFinite(med) ? +med.toFixed(4) : null, isFinite(iqr) ? +iqr.toFixed(4) : null] });
      if (b >= nF) break;
    }
    return { win: out, thr };
  }
  // ---- clustering ----
  function zNorm(vecs) {
    const D = vecs[0].length, mu = new Float64Array(D), sd = new Float64Array(D), cnt = new Float64Array(D);
    for (const v of vecs) for (let k = 0; k < D; k++) if (v[k] != null && isFinite(v[k])) { mu[k] += v[k]; cnt[k]++; }
    for (let k = 0; k < D; k++) mu[k] = cnt[k] ? mu[k] / cnt[k] : 0;
    for (const v of vecs) for (let k = 0; k < D; k++) if (v[k] != null && isFinite(v[k])) { const d = v[k] - mu[k]; sd[k] += d * d; }
    for (let k = 0; k < D; k++) sd[k] = Math.sqrt(sd[k] / Math.max(1, cnt[k])) || 1;
    const nc = (D - 2) / 2, W = new Float64Array(D);
    const P=api.P; for (let k = 0; k < nc; k++) W[k] = 1; for (let k = nc; k < 2 * nc; k++) W[k] = P.sd; W[D - 2] = P.pitch; W[D - 1] = P.iqr;
    return vecs.map(v => { const o = new Float64Array(D); for (let k = 0; k < D; k++) o[k] = (v[k] != null && isFinite(v[k])) ? (v[k] - mu[k]) / sd[k] * W[k] : 0; return o; });
  }
  const d2 = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) { const d = a[k] - b[k]; s += d * d; } return s; };
  function wardTree(X) {
    const n = X.length; let r2 = 0.3 * X[0].length, G;
    for (; ;) {
      G = []; for (let i = 0; i < n; i++) {
        let bg = -1, bd = Infinity; for (let g = 0; g < G.length; g++) { const d = d2(X[i], G[g].c); if (d < bd) { bd = d; bg = g; } }
        if (bg >= 0 && bd < r2) { const g = G[bg]; g.m.push(i); for (let k = 0; k < g.c.length; k++) g.c[k] += (X[i][k] - g.c[k]) / g.m.length; } else G.push({ m: [i], c: Float64Array.from(X[i]) });
      }
      if (G.length <= 300) break; r2 *= 1.5;
    }
    const cl = G.map(g => ({ c: Float64Array.from(g.c), n: g.m.length, alive: true }));
    const merges = []; let alive = cl.length;
    const cost = (a, b) => a.n * b.n / (a.n + b.n) * d2(a.c, b.c);
    while (alive > 1) {
      let bi = -1, bj = -1, bc = Infinity;
      for (let i = 0; i < cl.length; i++) { if (!cl[i].alive) continue; for (let j = i + 1; j < cl.length; j++) { if (!cl[j].alive) continue; const c = cost(cl[i], cl[j]); if (c < bc) { bc = c; bi = i; bj = j; } } }
      const A = cl[bi], B = cl[bj]; merges.push([bi, bj, bc]); const nn = A.n + B.n; for (let k = 0; k < A.c.length; k++) A.c[k] = (A.c[k] * A.n + B.c[k] * B.n) / nn; A.n = nn; B.alive = false; alive--;
    }
    return { groups: G.map(g => g.m), merges, n };
  }
  function cutTree(tree, k) {
    const G = tree.groups.map(m => m.slice()), alive = G.map(() => true); let cnt = G.length;
    for (const [i, j] of tree.merges) { if (cnt <= k) break; G[i] = G[i].concat(G[j]); alive[j] = false; cnt--; }
    const lab = new Int32Array(tree.n); let c = 0; G.forEach((m, i) => { if (!alive[i]) return; m.forEach(x => lab[x] = c); c++; }); return lab;
  }
  // k-means refinement starting from a labelling (improves Ward's greedy result)
  function refine(X, lab, k, iters) {
    const D = X[0].length; let L = Int32Array.from(lab);
    for (let it = 0; it < iters; it++) {
      const C = Array.from({ length: k }, () => new Float64Array(D)), n = new Float64Array(k);
      L.forEach((l, i) => { n[l]++; for (let d = 0; d < D; d++) C[l][d] += X[i][d]; });
      C.forEach((c, l) => { if (n[l]) for (let d = 0; d < D; d++) c[d] /= n[l]; });
      let ch = 0; for (let i = 0; i < X.length; i++) { let bl = L[i], bd = Infinity; for (let l = 0; l < k; l++) { if (!n[l]) continue; const d = d2(X[i], C[l]); if (d < bd) { bd = d; bl = l; } } if (bl !== L[i]) { L[i] = bl; ch++; } }
      if (!ch) break;
    }
    return L;
  }
  function silhouette(X, lab, k) {
    const n = X.length, idx = []; const step = Math.max(1, Math.floor(n / 500)); for (let i = 0; i < n; i += step) idx.push(i);
    let s = 0, m = 0; for (const i of idx) {
      const sum = new Float64Array(k), c = new Float64Array(k); for (const j of idx) { if (i === j) continue; const d = Math.sqrt(d2(X[i], X[j])); sum[lab[j]] += d; c[lab[j]]++; }
      const own = lab[i]; if (c[own] === 0) continue; const a = sum[own] / c[own]; let b = Infinity; for (let q = 0; q < k; q++) if (q !== own && c[q]) b = Math.min(b, sum[q] / c[q]); if (isFinite(b)) { s += (b - a) / Math.max(a, b); m++; }
    }
    return m ? s / m : 0;
  }
  function smooth(lab, times) { // majority filter over neighbours (5 windows)
    const out = Int32Array.from(lab);
    for (let i = 0; i < lab.length; i++) {
      const cnt = new Map(); for (let j = Math.max(0, i - 2); j <= Math.min(lab.length - 1, i + 2); j++) { if (Math.abs(times[j] - times[i]) > 3.1) continue; cnt.set(lab[j], (cnt.get(lab[j]) || 0) + (j === i ? 1.5 : 1)); }
      let bl = lab[i], bv = -1; for (const [l, v] of cnt) if (v > bv) { bv = v; bl = l; } out[i] = bl;
    }
    return out;
  }
  
  // win: [{t,f}] ; K: 0 = auto
  function cluster(win, K, verbose) {
    const n = win.length; if (n < 4 || K === 1) return { lab: new Int32Array(n), k: 1, scores: [] };
    const X = zNorm(win.map(w => w.f)), tree = wardTree(X), scores = [];
    let k = K;
    if (!k) {
      const kmax = Math.min(8, Math.floor(n / 6)), sc = [];
      for (let q = 2; q <= kmax; q++) {
        const lab = refine(X, cutTree(tree, q), q, 15); const sz = new Int32Array(q); lab.forEach(l => sz[l]++);
        const s = Math.min(...sz) < Math.max(3, n * 0.03) ? -1 : silhouette(X, lab, q); sc.push([q, s]); scores.push(s);
      }
      const best = Math.max(-1, ...sc.map(x => x[1]));
      k = best < api.P.sil ? 1 : sc.find(x => x[1] >= best - api.P.tol)[0];
    }
    k = Math.min(k, n);
    let lab = k > 1 ? refine(X, cutTree(tree, k), k, 20) : new Int32Array(n);
    lab = smooth(lab, win.map(w => w.t));
    return { lab, k, scores };
  }
  const api = { windows, cluster, WIN, STEP, P: { sd: 1, pitch: 3, iqr: 0, sil: 0.3, tol: 0.05 } };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.SPK = api;
})(typeof self !== 'undefined' ? self : this);
