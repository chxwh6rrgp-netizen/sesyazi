// Memory-lean audio decoding: reads m4a/mp4/mov, mp3 and wav in ~5 minute pieces
// (instead of decoding the whole file at once) and returns 16 kHz mono Int16 PCM.
// Unknown formats fall back to decoding the whole file.
(function (root) {
  "use strict";
  const SR = 16000, PIECE = 300; // seconds per decoded piece

  const OAC = () => root.OfflineAudioContext || root.webkitOfflineAudioContext;
  async function decodeBuf(buf) {
    const off = new (OAC())(1, SR, SR);
    return await new Promise((res, rej) => { try { const p = off.decodeAudioData(buf, res, rej); if (p && p.then) p.then(res, rej); } catch (e) { rej(e); } });
  }
  // AudioBuffer -> mono Float32 at 16 kHz (optionally through the clean-up filter chain)
  async function toMono(ab, clean) {
    if (clean) {
      const len = Math.max(1, Math.ceil(ab.duration * SR));
      const oc = new (OAC())(1, len, SR), s = oc.createBufferSource(); s.buffer = ab;
      const hp = oc.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 85;
      const hum = oc.createBiquadFilter(); hum.type = "notch"; hum.frequency.value = 50; hum.Q.value = 10;
      const lp = oc.createBiquadFilter(); lp.type = "lowpass"; lp.frequency.value = 7600;
      const pres = oc.createBiquadFilter(); pres.type = "peaking"; pres.frequency.value = 2800; pres.gain.value = 3;
      const comp = oc.createDynamicsCompressor(); comp.threshold.value = -30; comp.knee.value = 12; comp.ratio.value = 4; comp.attack.value = .005; comp.release.value = .2;
      s.connect(hp); hp.connect(hum); hum.connect(lp); lp.connect(pres); pres.connect(comp); comp.connect(oc.destination); s.start();
      return (await oc.startRendering()).getChannelData(0);
    }
    const n = ab.length, ch = ab.numberOfChannels; if (ch === 1) return ab.getChannelData(0);
    const out = new Float32Array(n), C = []; for (let c = 0; c < ch; c++) C.push(ab.getChannelData(c));
    for (let i = 0; i < n; i++) { let v = 0; for (let c = 0; c < ch; c++) v += C[c][i]; out[i] = v / ch; }
    return out;
  }
  // growable Int16 output
  function Sink(hintLen) {
    let a = new Int16Array(Math.max(SR, hintLen || SR * 60)), n = 0;
    return {
      put(f, at) { // write Float32 samples at sample position `at` (or append)
        const pos = at == null ? n : at, need = pos + f.length;
        if (need > a.length) { const b = new Int16Array(Math.max(need, Math.floor(a.length * 1.5))); b.set(a.subarray(0, n)); a = b; }
        for (let i = 0; i < f.length; i++) { const x = f[i]; a[pos + i] = x >= 1 ? 32767 : x <= -1 ? -32768 : (x * 32767) | 0; }
        n = Math.max(n, need);
      },
      get len() { return n; },
      done(total) { const L = total != null ? total : n; if (L > a.length) { const b = new Int16Array(L); b.set(a.subarray(0, n)); a = b; } return a.length === L ? a : a.slice(0, L); }
    };
  }
  const tick = () => new Promise(r => setTimeout(r, 0));
  const rd = async (blob, s, e) => new Uint8Array(await blob.slice(s, e).arrayBuffer());
  const u32 = (b, o) => ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0;
  const u16 = (b, o) => (b[o] << 8) | b[o + 1];
  const typ = (b, o) => String.fromCharCode(b[o], b[o + 1], b[o + 2], b[o + 3]);

  // ---------------- MP4 / M4A / MOV (AAC) ----------------
  async function findTopBox(blob, want) {
    let pos = 0; const size = blob.size;
    while (pos + 8 <= size) {
      const h = await rd(blob, pos, pos + 16); let len = u32(h, 0); const t = typ(h, 4); let hdr = 8;
      if (len === 1) { len = u32(h, 8) * 4294967296 + u32(h, 12); hdr = 16; } else if (len === 0) len = size - pos;
      if (len < 8) return null;
      if (t === want) return { pos, len, hdr };
      pos += len;
    }
    return null;
  }
  function boxes(b, s, e) { // child boxes inside [s,e)
    const out = []; let p = s;
    while (p + 8 <= e) { let len = u32(b, p); const t = typ(b, p + 4); let hdr = 8; if (len === 1) { len = u32(b, p + 8) * 4294967296 + u32(b, p + 12); hdr = 16; } else if (len === 0) len = e - p; if (len < 8 || p + len > e) break; out.push({ t, s: p + hdr, e: p + len }); p += len; }
    return out;
  }
  const child = (b, box, t) => boxes(b, box.s, box.e).find(x => x.t === t);
  function findDeep(b, s, e, t, depth) { if (depth > 6) return null; for (const x of boxes(b, s, e)) { if (x.t === t) return x; const r = findDeep(b, x.s, x.e, t, depth + 1); if (r) return r; } return null; }
  function parseEsds(b, box) { // returns AudioSpecificConfig bytes
    let p = box.s + 4; const end = box.e;
    const desc = () => { const tag = b[p++]; let len = 0, k = 0; while (k < 4) { const c = b[p++]; len = (len << 7) | (c & 0x7f); k++; if (!(c & 0x80)) break; } return { tag, len, start: p }; };
    while (p < end) {
      const d = desc();
      if (d.tag === 0x03) { const flags = b[p + 2]; p += 3; if (flags & 0x80) p += 2; if (flags & 0x40) p += 1 + b[p]; if (flags & 0x20) p += 2; continue; }
      if (d.tag === 0x04) { p += 13; continue; }
      if (d.tag === 0x05) return b.slice(d.start, d.start + d.len);
      p = d.start + d.len;
    }
    return null;
  }
  const SFREQ = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];
  function parseAsc(a) {
    let bit = 0; const get = n => { let v = 0; for (let i = 0; i < n; i++) { v = (v << 1) | ((a[bit >> 3] >> (7 - (bit & 7))) & 1); bit++; } return v; };
    const aotf = () => { let t = get(5); if (t === 31) t = 32 + get(6); return t; };
    const srf = () => { let i = get(4); if (i === 15) { const f = get(24); i = SFREQ.reduce((bi, v, k) => Math.abs(v - f) < Math.abs(SFREQ[bi] - f) ? k : bi, 0); } return i; };
    let aot = aotf(), sfi = srf(); const ch = get(4);
    if (aot === 5 || aot === 29) { srf(); aot = aotf(); } // HE-AAC: core layer is described next
    if (aot < 1 || aot > 4) aot = 2;
    return { aot, sfi, ch: ch || 1 };
  }
  async function decodeMp4(blob, clean, onProg) {
    const mb = await findTopBox(blob, "moov"); if (!mb || mb.len > 200e6) return null;
    if (await findTopBox(blob, "moof")) return null; // fragmented mp4: let the browser handle it
    const b = await rd(blob, mb.pos, mb.pos + mb.len), moov = { s: mb.hdr, e: mb.len };
    let tr = null;
    for (const t of boxes(b, moov.s, moov.e).filter(x => x.t === "trak")) {
      const mdia = child(b, t, "mdia"); if (!mdia) continue; const hdlr = child(b, mdia, "hdlr");
      if (hdlr && typ(b, hdlr.s + 8) === "soun") { tr = { t, mdia }; break; }
    }
    if (!tr) return null;
    const mdhd = child(b, tr.mdia, "mdhd"); const v = b[mdhd.s]; const timescale = v === 1 ? u32(b, mdhd.s + 20) : u32(b, mdhd.s + 12);
    const stbl = findDeep(b, tr.mdia.s, tr.mdia.e, "stbl", 0); if (!stbl) return null;
    const stsd = child(b, stbl, "stsd"); const entry = boxes(b, stsd.s + 8, stsd.e)[0]; if (!entry || entry.t !== "mp4a") return null;
    const ver = u16(b, entry.s + 8); const kids = entry.s + 28 + (ver === 1 ? 16 : ver === 2 ? 36 : 0);
    const esds = findDeep(b, kids, entry.e, "esds", 0); if (!esds) return null;
    const ascB = parseEsds(b, esds); if (!ascB || !ascB.length) return null;
    const asc = parseAsc(ascB);
    // sample table
    const stsz = child(b, stbl, "stsz"), stsc = child(b, stbl, "stsc"), stts = child(b, stbl, "stts"), stco = child(b, stbl, "stco") || child(b, stbl, "co64");
    if (!stsz || !stsc || !stts || !stco) return null;
    const fixed = u32(b, stsz.s + 4), count = u32(b, stsz.s + 8);
    const size = i => fixed || u32(b, stsz.s + 12 + 4 * i);
    const nChunks = u32(b, stco.s + 4), co64 = stco.t === "co64";
    const chunkOff = i => co64 ? u32(b, stco.s + 8 + 8 * i) * 4294967296 + u32(b, stco.s + 12 + 8 * i) : u32(b, stco.s + 8 + 4 * i);
    const offs = new Float64Array(count); { // sample -> file offset
      const runs = u32(b, stsc.s + 4); let si = 0;
      for (let r = 0; r < runs; r++) {
        const first = u32(b, stsc.s + 8 + 12 * r) - 1, per = u32(b, stsc.s + 12 + 12 * r);
        const next = r + 1 < runs ? u32(b, stsc.s + 8 + 12 * (r + 1)) - 1 : nChunks;
        for (let c = first; c < next && si < count; c++) { let o = chunkOff(c); for (let k = 0; k < per && si < count; k++) { offs[si] = o; o += size(si); si++; } }
      }
      if (si < count) return null;
    }
    const tim = new Float64Array(count + 1); { // sample -> start time (s)
      const ent = u32(b, stts.s + 4); let si = 0, t = 0;
      for (let k = 0; k < ent; k++) { const c = u32(b, stts.s + 8 + 8 * k), d = u32(b, stts.s + 12 + 8 * k); for (let j = 0; j < c && si < count; j++) { tim[si++] = t / timescale; t += d; } }
      while (si <= count) tim[si++] = t / timescale;
    }
    const total = Math.round(tim[count] * SR), sink = Sink(total);
    for (let a = 0; a < count;) {
      let e = a; while (e < count && tim[e] - tim[a] < PIECE) e++;
      // gather frames (contiguous runs read in one go)
      const parts = []; let bytes = 0;
      for (let i = a; i < e;) { let j = i + 1; while (j < e && offs[j] === offs[j - 1] + size(j - 1) && offs[j] - offs[i] < 8e6) j++;
        const raw = await rd(blob, offs[i], offs[j - 1] + size(j - 1)); let p = 0;
        for (let k = i; k < j; k++) { const L = size(k) + 7, h = new Uint8Array(L);
          h[0] = 0xFF; h[1] = 0xF1; h[2] = (((asc.aot - 1) & 3) << 6) | ((asc.sfi & 15) << 2) | ((asc.ch >> 2) & 1); h[3] = ((asc.ch & 3) << 6) | ((L >> 11) & 3); h[4] = (L >> 3) & 0xFF; h[5] = ((L & 7) << 5) | 0x1F; h[6] = 0xFC;
          h.set(raw.subarray(p, p + size(k)), 7); p += size(k); parts.push(h); bytes += L; }
        i = j; }
      const adts = new Uint8Array(bytes); { let p = 0; for (const h of parts) { adts.set(h, p); p += h.length; } }
      parts.length = 0;
      const mono = await toMono(await decodeBuf(adts.buffer), clean);
      // place at the piece's exact start time; trim/pad to its exact length (no drift)
      const at = Math.round(tim[a] * SR), want = Math.round(tim[e] * SR) - at;
      sink.put(mono.length > want ? mono.subarray(0, want) : mono, at);
      a = e; onProg && onProg(tim[a] / tim[count]); await tick();
    }
    return sink.done(total);
  }

  // ---------------- MP3 ----------------
  const BR = { 1: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320], 2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160] };
  const SRT = { 3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000] };
  function mp3Frame(b, i) { // frame length at i or 0
    if (b[i] !== 0xFF || (b[i + 1] & 0xE0) !== 0xE0) return 0;
    const ver = (b[i + 1] >> 3) & 3, layer = (b[i + 1] >> 1) & 3, bri = b[i + 2] >> 4, sri = (b[i + 2] >> 2) & 3, pad = (b[i + 2] >> 1) & 1;
    if (ver === 1 || layer !== 1 || bri === 0 || bri === 15 || sri === 3) return 0; // layer III only
    const br = BR[ver === 3 ? 1 : 2][bri] * 1000, sr = SRT[ver][sri];
    return Math.floor((ver === 3 ? 144 : 72) * br / sr) + pad;
  }
  async function decodeMp3(blob, clean, onProg) {
    const head = await rd(blob, 0, 10); let start = 0;
    if (head[0] === 0x49 && head[1] === 0x44 && head[2] === 0x33) start = 10 + ((head[6] & 127) << 21 | (head[7] & 127) << 14 | (head[8] & 127) << 7 | (head[9] & 127));
    const size = blob.size, sink = Sink(); let pos = start, guard = 0;
    // ~5 min per piece at the file's average bitrate is unknown: use 5 MB pieces (~4-5 min at 128 kbps)
    const STEP = 5 * 1048576;
    while (pos < size && guard++ < 100000) {
      let end = Math.min(size, pos + STEP);
      if (end < size) { const w = await rd(blob, end, Math.min(size, end + 8192)); let k = 0, ok = -1;
        for (; k < w.length - 4; k++) { const L = mp3Frame(w, k); if (L && (k + L + 2 > w.length || mp3Frame(w, k + L))) { ok = k; break; } }
        end = ok >= 0 ? end + ok : Math.min(size, end + 8192); }
      const mono = await toMono(await decodeBuf((await rd(blob, pos, end)).buffer), clean);
      sink.put(mono); pos = end; onProg && onProg(pos / size); await tick();
    }
    return sink.done();
  }

  // ---------------- WAV ----------------
  async function decodeWav(blob, clean, onProg) {
    const h = await rd(blob, 0, Math.min(blob.size, 65536)); let p = 12, fmt = null, data = null;
    const le32 = o => (h[o] | h[o + 1] << 8 | h[o + 2] << 16 | h[o + 3] << 24) >>> 0;
    while (p + 8 <= h.length) { const id = typ(h, p), len = le32(p + 4); if (id === "fmt ") fmt = h.slice(p, p + 8 + len); if (id === "data") { data = { s: p + 8, len: Math.min(len, blob.size - p - 8) }; break; } p += 8 + len + (len & 1); }
    if (!fmt || !data) return null;
    const ba = fmt[20] | fmt[21] << 8, rate = (fmt[12] | fmt[13] << 8 | fmt[14] << 16 | fmt[15] << 24) >>> 0;
    if (!ba || !rate) return null;
    const step = Math.max(ba, Math.floor(rate * PIECE / 1) * ba), sink = Sink();
    for (let o = 0; o < data.len; o += step) {
      const n = Math.min(step, data.len - o), body = await rd(blob, data.s + o, data.s + o + n);
      const w = new Uint8Array(8 + fmt.length + 4 + 8 + n); const dv = new DataView(w.buffer);
      w.set([82, 73, 70, 70]); dv.setUint32(4, w.length - 8, true); w.set([87, 65, 86, 69], 8); w.set(fmt, 12);
      let q = 12 + fmt.length; w.set([100, 97, 116, 97], q); dv.setUint32(q + 4, n, true); w.set(body, q + 8);
      sink.put(await toMono(await decodeBuf(w.buffer), clean)); onProg && onProg((o + n) / data.len); await tick();
    }
    return sink.done();
  }

  async function decodeWhole(blob, clean) {
    const mono = await toMono(await decodeBuf(await blob.arrayBuffer()), clean);
    const s = Sink(mono.length); s.put(mono); return s.done();
  }
  function normalize(a) { // lift quiet recordings (in place)
    let peak = 0; for (let i = 0; i < a.length; i += 3) { const v = Math.abs(a[i]); if (v > peak) peak = v; }
    const g = peak > 30 ? Math.min(30, 0.92 * 32767 / peak) : 1;
    if (g > 1.05) for (let i = 0; i < a.length; i++) { const x = a[i] * g; a[i] = x > 32767 ? 32767 : x < -32768 ? -32768 : x; }
    return a;
  }
  // main entry
  async function decode(blob, clean, onProg) {
    const h = await rd(blob, 0, 12); let out = null, how = "whole";
    try {
      if (typ(h, 4) === "ftyp") { out = await decodeMp4(blob, clean, onProg); how = "mp4"; }
      else if ((h[0] === 0x49 && h[1] === 0x44 && h[2] === 0x33) || (h[0] === 0xFF && (h[1] & 0xE0) === 0xE0)) { out = await decodeMp3(blob, clean, onProg); how = "mp3"; }
      else if (typ(h, 0) === "RIFF" && typ(h, 8) === "WAVE") { out = await decodeWav(blob, clean, onProg); how = "wav"; }
    } catch (e) { out = null; }
    if (!out || out.length < SR / 2) { out = await decodeWhole(blob, clean); how = "whole"; }
    decode.last = how;
    return normalize(out);
  }
  root.AUDIO = { decode, SR };
})(typeof self !== "undefined" ? self : this);
