function hoasegDefine() {
  "use strict";

  var SIZE = 180;
  var PAD = 6;
  var S0 = SIZE + 2 * PAD;
  var CH = [32, 64, 128, 256, 512];
  var SZ = [192, 96, 48, 24, 12];
  var NCLS = 3;
  var NPIX = SIZE * SIZE;
  var VALUES = 7759587;

  function layerTable() {
    var L = [], cin = 1, i;
    for (i = 0; i <= 4; i++) {
      L.push({ kind: "conv3", cin: cin, cout: CH[i] });
      L.push({ kind: "conv3", cin: CH[i], cout: CH[i] });
      cin = CH[i];
    }
    for (i = 3; i >= 0; i--) {
      L.push({ kind: "up", cin: CH[i + 1], cout: CH[i] });
      L.push({ kind: "conv3", cin: 2 * CH[i], cout: CH[i] });
      L.push({ kind: "conv3", cin: CH[i], cout: CH[i] });
    }
    L.push({ kind: "head", cin: CH[0], cout: NCLS });
    var n = 0;
    L.forEach(function (l) {
      l.nw = (l.kind === "conv3" ? 9 : l.kind === "up" ? 4 : 1) * l.cin * l.cout;
      n += l.nw + l.cout;
    });
    if (n !== VALUES) throw new Error("layer table does not match the weight file");
    return L;
  }

  function kernelWasm() {
    function u(n) { var o = []; do { var b = n & 0x7f; n >>>= 7; if (n) b |= 0x80; o.push(b); } while (n); return o; }
    function s(n) {
      var o = [];
      for (;;) {
        var b = n & 0x7f; n >>= 7;
        if ((n === 0 && !(b & 0x40)) || (n === -1 && (b & 0x40))) { o.push(b); return o; }
        o.push(b | 0x80);
      }
    }
    function str(t) { var o = [t.length]; for (var i = 0; i < t.length; i++) o.push(t.charCodeAt(i)); return o; }
    function sec(id, body) { return [id].concat(u(body.length), body); }
    function cat() { var o = []; for (var i = 0; i < arguments.length; i++) o = o.concat(arguments[i]); return o; }
    var I32 = 0x7f, F32 = 0x7d, V128 = 0x7b;

    function get(i) { return [0x20, i]; }
    function set(i) { return [0x21, i]; }
    function tee(i) { return [0x22, i]; }
    function c(v) { return [0x41].concat(s(v)); }
    var add = [0x6a], sub = [0x6b], mul = [0x6c], ltu = [0x49];
    function vload(off) { return [0xfd, 0x00, 0x02].concat(u(off)); }
    var vmul = [0xfd, 0xe6, 0x01], vadd = [0xfd, 0xe4, 0x01];
    var vzero = [0xfd, 0x0c, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    function lane(i) { return [0xfd, 0x1f, i]; }
    var fload = [0x2a, 0x02, 0x00], fstore = [0x38, 0x02, 0x00], fadd = [0x92], fmax = [0x97];
    var loop = [0x03, 0x40], end = [0x0b], again = [0x0d, 0x00];
    function acc(a, off) { return cat(get(a), get(16), vload(off), get(17), vload(off), vmul, vadd, set(a)); }
    var code = cat(
      c(0), set(12),
      loop,
        get(6), get(12), c(4), mul, add, fload, set(20),
        get(0), set(15),
        get(7), get(12), c(4), mul, add, set(19),
        c(0), set(13),
        loop,
          vzero, set(21), vzero, set(22), vzero, set(23), vzero, set(24),
          get(15), set(16),
          get(5), set(17),
          c(0), set(14),
          loop,
            get(16), get(3), add, set(18),
            loop,
              acc(21, 0), acc(22, 16), acc(23, 32), acc(24, 48),
              get(17), c(64), add, set(17),
              get(16), c(64), add, tee(16),
              get(18), ltu, again,
            end,
            get(16), get(3), sub, get(1), add, set(16),
            get(14), c(1), add, tee(14), get(4), ltu, again,
          end,
          get(19),
          get(21), get(22), vadd, get(23), get(24), vadd, vadd, set(21),
          get(21), lane(0), get(21), lane(1), fadd, get(21), lane(2), get(21), lane(3), fadd, fadd,
          get(20), fadd, get(11), fmax,
          fstore,
          get(15), get(2), add, set(15),
          get(19), get(8), add, set(19),
          get(13), c(1), add, tee(13), get(9), ltu, again,
        end,
        get(5), get(3), get(4), mul, add, set(5),
        get(12), c(1), add, tee(12), get(10), ltu, again,
      end,
      end
    );
    var body = cat([3, 8, I32, 1, F32, 4, V128], code);
    var type = [1, 0x60, 12, I32, I32, I32, I32, I32, I32, I32, I32, I32, I32, I32, F32, 0];
    var imp = cat([1], str("env"), str("mem"), [0x02, 0x00, 0x01]);
    var bytes = cat(
      [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00],
      sec(1, type), sec(2, imp), sec(3, [1, 0]), sec(7, cat([1], str("conv"), [0x00, 0x00])), sec(10, cat([1], u(body.length), body))
    );
    return new Uint8Array(bytes);
  }

  function kernelJS(F) {
    return function (inp, rowB, pixB, n, nRows, w, b, out, outPixB, width, cout, lo) {
      var nf = n >> 2, rowF = rowB >> 2, pixF = pixB >> 2, outF = outPixB >> 2, wo = w >> 2, b0 = b >> 2;
      for (var o = 0; o < cout; o++) {
        var bias = F[b0 + o], p = inp >> 2, op = (out >> 2) + o;
        for (var x = 0; x < width; x++) {
          var s0 = 0, s1 = 0, s2 = 0, s3 = 0, q = p, wq = wo;
          for (var r = 0; r < nRows; r++) {
            for (var e = q + nf; q < e; q += 4, wq += 4) {
              s0 += F[q] * F[wq]; s1 += F[q + 1] * F[wq + 1]; s2 += F[q + 2] * F[wq + 2]; s3 += F[q + 3] * F[wq + 3];
            }
            q += rowF - nf;
          }
          var v = s0 + s1 + (s2 + s3) + bias;
          F[op] = v > lo ? v : lo;
          p += pixF; op += outF;
        }
        wo += nf * nRows;
      }
    };
  }

  function bytesFromBase64(s) {
    if (typeof Uint8Array.fromBase64 === "function") return Uint8Array.fromBase64(s);
    if (typeof atob === "function") {
      var bin = atob(s), out = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }
    return new Uint8Array(Buffer.from(s, "base64"));
  }

  var HALF = null;
  function halfTable() {
    if (HALF) return HALF;
    var t = new Float32Array(65536);
    for (var h = 0; h < 65536; h++) {
      var sign = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
      t[h] = e === 0 ? sign * f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : sign * Infinity) : sign * (1 + f / 1024) * Math.pow(2, e - 15);
    }
    return (HALF = t);
  }

  function normalize(u8) {
    var n = 0, sum = 0, i, v;
    for (i = 0; i < NPIX; i++) { v = u8[i]; if (v > 10) { n++; sum += v / 255; } }
    if (n < 1) n = 1;
    var mean = sum / n, ss = 0;
    for (i = 0; i < NPIX; i++) { v = u8[i]; if (v > 10) { var d = v / 255 - mean; ss += d * d; } }
    var sd = Math.sqrt(ss / n);
    if (sd < 0.02) sd = 0.02;
    var out = new Float32Array(NPIX);
    for (i = 0; i < NPIX; i++) out[i] = (u8[i] / 255 - mean) / sd;
    return out;
  }

  function Net(options) {
    options = options || {};
    var L = (this.layers = layerTable());
    var off = 64;
    function alloc(floats) { var o = off; off += Math.ceil((floats * 4) / 64) * 64; return o; }
    function map(S, C) { return alloc((S + 2) * (S + 2) * C); }
    L.forEach(function (l) { l.wp = alloc(l.nw); l.bp = alloc(l.cout); });
    this.in0 = map(S0, 1);
    this.A = []; this.E = []; this.P = []; this.CAT = [];
    for (var i = 0; i <= 4; i++) {
      this.A[i] = map(SZ[i], CH[i]);
      this.E[i] = map(SZ[i], CH[i]);
      if (i > 0) this.P[i] = map(SZ[i], CH[i - 1]);
      if (i < 4) this.CAT[i] = map(SZ[i], 2 * CH[i]);
    }
    this.logitsA = alloc(S0 * S0 * NCLS);
    this.logitsB = alloc(S0 * S0 * NCLS);
    var pages = Math.ceil(off / 65536) + 1;
    this.simd = false;
    if (options.simd !== false && typeof WebAssembly === "object") {
      try {
        var mem = new WebAssembly.Memory({ initial: pages });
        var inst = new WebAssembly.Instance(new WebAssembly.Module(kernelWasm()), { env: { mem: mem } });
        this.F = new Float32Array(mem.buffer);
        this.kernel = inst.exports.conv;
        this.simd = true;
      } catch (e) {
        this.simd = false;
      }
    }
    if (!this.simd) {
      this.F = new Float32Array(pages * 16384);
      this.kernel = kernelJS(this.F);
    }
    this.loaded = false;
  }

  Net.prototype.loadHalfBytes = function (bytes) {
    if (bytes.length !== 2 * VALUES) throw new Error("weight file has " + bytes.length + " bytes, expected " + 2 * VALUES);
    if (bytes.byteOffset % 2) bytes = bytes.slice();
    var h = new Uint16Array(bytes.buffer, bytes.byteOffset, VALUES), T = halfTable(), F = this.F, k = 0;
    this.layers.forEach(function (l) {
      var i, d = l.wp >> 2;
      for (i = 0; i < l.nw; i++) F[d + i] = T[h[k++]];
      d = l.bp >> 2;
      for (i = 0; i < l.cout; i++) F[d + i] = T[h[k++]];
    });
    this.loaded = true;
  };

  Net.prototype.loadBase64 = function (s) { this.loadHalfBytes(bytesFromBase64(s)); };

  Net.prototype._conv3 = function (src, dst, S, cin, cout, l) {
    var rowB = (S + 2) * cin * 4, outRowB = (S + 2) * cout * 4, k = this.kernel;
    for (var y = 0; y < S; y++) k(src + y * rowB, rowB, cin * 4, 3 * cin * 4, 3, l.wp, l.bp, dst + (y + 1) * outRowB + cout * 4, cout * 4, S, cout, 0);
  };

  Net.prototype._conv3first = function (src, dst, l) {
    var F = this.F, S = S0, W2 = S + 2, cout = l.cout, s0 = src >> 2, d0 = dst >> 2, w0 = l.wp >> 2, b0 = l.bp >> 2;
    for (var y = 0; y < S; y++) {
      for (var x = 0; x < S; x++) {
        var p = s0 + y * W2 + x;
        var a0 = F[p], a1 = F[p + 1], a2 = F[p + 2], a3 = F[p + W2], a4 = F[p + W2 + 1], a5 = F[p + W2 + 2];
        var a6 = F[p + 2 * W2], a7 = F[p + 2 * W2 + 1], a8 = F[p + 2 * W2 + 2];
        var d = d0 + ((y + 1) * W2 + x + 1) * cout;
        for (var o = 0; o < cout; o++) {
          var w = w0 + o * 9;
          var v = a0 * F[w] + a1 * F[w + 1] + a2 * F[w + 2] + a3 * F[w + 3] + a4 * F[w + 4] + a5 * F[w + 5] + a6 * F[w + 6] + a7 * F[w + 7] + a8 * F[w + 8] + F[b0 + o];
          F[d + o] = v > 0 ? v : 0;
        }
      }
    }
  };

  Net.prototype._pool = function (src, dst, S, C) {
    var F = this.F, h = S >> 1, sw = (S + 2) * C, dw = (h + 2) * C, s0 = src >> 2, d0 = dst >> 2;
    for (var y = 0; y < h; y++) {
      var a = s0 + (2 * y + 1) * sw + C, b = a + sw, d = d0 + (y + 1) * dw + C;
      for (var x = 0; x < h; x++) {
        for (var c = 0; c < C; c++) {
          var v0 = F[a + c], v1 = F[a + C + c], v2 = F[b + c], v3 = F[b + C + c];
          var m0 = v0 > v1 ? v0 : v1, m1 = v2 > v3 ? v2 : v3;
          F[d + c] = m0 > m1 ? m0 : m1;
        }
        a += 2 * C; b += 2 * C; d += C;
      }
    }
  };

  Net.prototype._up = function (src, cat, s, cin, cout, l) {
    var k = this.kernel, S = 2 * s, inRowB = (s + 2) * cin * 4, catPixB = 2 * cout * 4, catRowB = (S + 2) * catPixB, wq = cin * cout * 4;
    for (var y = 0; y < s; y++) {
      var ip = src + (y + 1) * inRowB + cin * 4;
      for (var ky = 0; ky < 2; ky++) {
        for (var kx = 0; kx < 2; kx++) {
          k(ip, 0, cin * 4, cin * 4, 1, l.wp + (ky * 2 + kx) * wq, l.bp, cat + (2 * y + ky + 1) * catRowB + (kx + 1) * catPixB, 2 * catPixB, s, cout, -Infinity);
        }
      }
    }
  };

  Net.prototype._skip = function (e, cat, S, C) {
    var F = this.F, ew = (S + 2) * C, cw = (S + 2) * 2 * C, e0 = e >> 2, c0 = cat >> 2;
    for (var y = 1; y <= S; y++) {
      var a = e0 + y * ew + C, d = c0 + y * cw + 3 * C;
      for (var x = 0; x < S; x++) { F.copyWithin(d, a, a + C); a += C; d += 2 * C; }
    }
  };

  Net.prototype._head = function (src, out, l) {
    var k = this.kernel, cin = l.cin, rowB = (S0 + 2) * cin * 4;
    for (var y = 0; y < S0; y++) k(src + (y + 1) * rowB + cin * 4, 0, cin * 4, cin * 4, 1, l.wp, l.bp, out + y * S0 * NCLS * 4, NCLS * 4, S0, NCLS, -Infinity);
  };

  Net.prototype._forward = function (out) {
    var L = this.layers, li = 0, src = this.in0, cin = 1, i, S, C;
    for (i = 0; i <= 4; i++) {
      S = SZ[i]; C = CH[i];
      if (i === 0) this._conv3first(src, this.A[0], L[li++]);
      else this._conv3(src, this.A[i], S, cin, C, L[li++]);
      this._conv3(this.A[i], this.E[i], S, C, C, L[li++]);
      if (i < 4) { this._pool(this.E[i], this.P[i + 1], S, C); src = this.P[i + 1]; cin = C; }
    }
    var x = this.E[4];
    for (i = 3; i >= 0; i--) {
      S = SZ[i]; C = CH[i];
      this._up(x, this.CAT[i], SZ[i + 1], CH[i + 1], C, L[li++]);
      this._skip(this.E[i], this.CAT[i], S, C);
      this._conv3(this.CAT[i], this.A[i], S, 2 * C, C, L[li++]);
      this._conv3(this.A[i], this.E[i], S, C, C, L[li++]);
      x = this.E[i];
    }
    this._head(x, out, L[li++]);
  };

  Net.prototype._setInput = function (xn, flip) {
    var F = this.F, base = this.in0 >> 2, W2 = S0 + 2;
    for (var y = 0; y < SIZE; y++) {
      var d = base + (y + PAD + 1) * W2 + PAD + 1, r = y * SIZE;
      if (flip) for (var x = 0; x < SIZE; x++) F[d + x] = xn[r + SIZE - 1 - x];
      else for (var x2 = 0; x2 < SIZE; x2++) F[d + x2] = xn[r + x2];
    }
  };

  Net.prototype.run = function (u8) {
    if (!this.loaded) throw new Error("the model has no weights yet");
    var xn = normalize(u8), F = this.F;
    this._setInput(xn, false);
    this._forward(this.logitsA);
    this._setInput(xn, true);
    this._forward(this.logitsB);
    var out = new Float32Array(NCLS * NPIX), a = this.logitsA >> 2, b = this.logitsB >> 2;
    for (var y = 0; y < SIZE; y++) {
      for (var x = 0; x < SIZE; x++) {
        var p = a + ((y + PAD) * S0 + x + PAD) * NCLS, q = b + ((y + PAD) * S0 + (S0 - 1 - x - PAD)) * NCLS;
        var l0 = (F[p] + F[q]) / 2, l1 = (F[p + 1] + F[q + 1]) / 2, l2 = (F[p + 2] + F[q + 2]) / 2;
        var m = l0 > l1 ? (l0 > l2 ? l0 : l2) : l1 > l2 ? l1 : l2;
        var e0 = Math.exp(l0 - m), e1 = Math.exp(l1 - m), e2 = Math.exp(l2 - m), t = e0 + e1 + e2, i = y * SIZE + x;
        out[i] = e0 / t; out[NPIX + i] = e1 / t; out[2 * NPIX + i] = e2 / t;
      }
    }
    return out;
  };

  function argmax(probs) {
    var lab = new Uint8Array(NPIX);
    for (var i = 0; i < NPIX; i++) {
      var c = 0, m = probs[i];
      if (probs[NPIX + i] > m) { c = 1; m = probs[NPIX + i]; }
      if (probs[2 * NPIX + i] > m) c = 2;
      lab[i] = c;
    }
    return lab;
  }

  function largestFilled(lab, cls, W, H) {
    var N = W * H, comp = new Int32Array(N), stack = new Int32Array(N), best = 0, bestArea = 0, n = 0, i, j, x, y, sp;
    for (i = 0; i < N; i++) {
      if (lab[i] !== cls || comp[i]) continue;
      n++;
      var area = 0;
      sp = 0; stack[sp++] = i; comp[i] = n;
      while (sp) {
        j = stack[--sp]; area++;
        x = j % W; y = (j - x) / W;
        for (var dy = -1; dy <= 1; dy++) {
          var yy = y + dy;
          if (yy < 0 || yy >= H) continue;
          for (var dx = -1; dx <= 1; dx++) {
            var xx = x + dx;
            if (xx < 0 || xx >= W) continue;
            var k = yy * W + xx;
            if (lab[k] === cls && !comp[k]) { comp[k] = n; stack[sp++] = k; }
          }
        }
      }
      if (area > bestArea) { bestArea = area; best = n; }
    }
    if (!n) return null;

    var out = new Uint8Array(N);
    sp = 0;
    function seed(k2) { if (comp[k2] !== best && !out[k2]) { out[k2] = 1; stack[sp++] = k2; } }
    for (x = 0; x < W; x++) { seed(x); seed((H - 1) * W + x); }
    for (y = 0; y < H; y++) { seed(y * W); seed(y * W + W - 1); }
    while (sp) {
      j = stack[--sp]; x = j % W; y = (j - x) / W;
      if (x > 0) seed(j - 1);
      if (x < W - 1) seed(j + 1);
      if (y > 0) seed(j - W);
      if (y < H - 1) seed(j + W);
    }
    for (i = 0; i < N; i++) out[i] = out[i] ? 0 : 1;
    return out;
  }

  function cleanup(lab, W, H) {
    W = W || SIZE; H = H || SIZE;
    var out = new Uint8Array(W * H);
    for (var cls = 1; cls <= 2; cls++) {
      var m = largestFilled(lab, cls, W, H);
      if (!m) continue;
      for (var i = 0; i < W * H; i++) if (m[i] && !out[i]) out[i] = cls;
    }
    return out;
  }

  function labelsFromSum(sum, count) {
    var raw = argmax(sum);
    var lab = cleanup(raw);
    var conf = 0, n = 0;
    for (var i = 0; i < NPIX; i++) if (lab[i]) { conf += sum[lab[i] * NPIX + i] / count; n++; }
    return { labels: lab, raw: raw, boneConfidence: n ? conf / n : 0 };
  }

  function workerBody(E) {
    var net = null;
    self.onmessage = function (ev) {
      var m = ev.data;
      try {
        if (m.type === "weights") {
          net = new E.Net({ simd: m.simd });
          net.loadBase64(m.b64);
          self.postMessage({ type: "loaded", simd: net.simd });
        } else if (m.type === "run") {
          var t0 = Date.now();
          var p = net.run(m.u8);
          self.postMessage({ type: "done", job: m.job, probs: p, ms: Date.now() - t0 }, [p.buffer]);
        }
      } catch (e) {
        self.postMessage({ type: "error", job: m.job, message: String((e && e.message) || e) });
      }
    };
  }

  var workerUrl = null;
  function makeWorker() {
    if (typeof Worker !== "function" || typeof Blob !== "function" || typeof URL !== "function") return null;
    try {
      if (!workerUrl) {
        var text = "var E = (" + hoasegDefine.toString() + ")();\n(" + workerBody.toString() + ")(E);\n";
        workerUrl = URL.createObjectURL(new Blob([text], { type: "text/javascript" }));
      }
      return new Worker(workerUrl);
    } catch (e) {
      return null;
    }
  }

  function Member(b64, options) {
    var self_ = this;
    this.simd = false;
    this.inWorker = false;
    this.pending = {};
    this.jobs = 0;
    this.ready = new Promise(function (resolve, reject) {
      var w = options.workers === false ? null : makeWorker();
      function inline() {
        try {
          var net = new Net({ simd: options.simd });
          net.loadBase64(b64);
          b64 = null;
          self_.net = net; self_.simd = net.simd;
          resolve(self_);
        } catch (e) { reject(e); }
      }
      if (!w) { setTimeout(inline, 0); return; }
      var started = false;
      w.onmessage = function (ev) {
        var m = ev.data;
        if (m.type === "loaded") { started = true; b64 = null; self_.worker = w; self_.inWorker = true; self_.simd = m.simd; resolve(self_); }
        else if (m.type === "done") { var p = self_.pending[m.job]; delete self_.pending[m.job]; if (p) p.resolve({ probs: m.probs, ms: m.ms }); }
        else if (m.type === "error") {
          if (!started) { w.terminate(); inline(); return; }
          var q = self_.pending[m.job]; delete self_.pending[m.job]; if (q) q.reject(new Error(m.message));
        }
      };
      w.onerror = function (ev) {
        if (ev && ev.preventDefault) ev.preventDefault();
        if (!started) { w.terminate(); inline(); }
      };
      w.postMessage({ type: "weights", b64: b64, simd: options.simd });
    });
  }

  Member.prototype.run = function (u8) {
    var self_ = this;
    if (this.worker) {
      return new Promise(function (resolve, reject) {
        var job = ++self_.jobs;
        self_.pending[job] = { resolve: resolve, reject: reject };
        self_.worker.postMessage({ type: "run", job: job, u8: u8 });
      });
    }
    return new Promise(function (resolve, reject) {
      setTimeout(function () {
        try { var t0 = Date.now(); var p = self_.net.run(u8); resolve({ probs: p, ms: Date.now() - t0 }); } catch (e) { reject(e); }
      }, 0);
    });
  };

  function Ensemble(options) {
    this.options = options || {};
    this.members = [];
    this.loading = [];
  }

  Ensemble.prototype.add = function (b64) {
    var self_ = this;
    var p = new Member(b64, this.options).ready.then(function (m) { self_.members.push(m); return m; });
    this.loading.push(p);
    return p;
  };

  Ensemble.prototype.segment = function (u8, onPart) {
    var self_ = this;
    return Promise.all(this.loading).then(function () {
      var ms = self_.members, total = ms.length, done = 0, t0 = Date.now();
      if (!total) throw new Error("no model is loaded");
      return Promise.all(ms.map(function (m) {
        return m.run(u8.slice()).then(function (r) { done++; if (onPart) onPart(done, total); return r; });
      })).then(function (parts) {
        var sum = new Float32Array(NCLS * NPIX);
        parts.forEach(function (r) { var p = r.probs; for (var i = 0; i < sum.length; i++) sum[i] += p[i]; });
        var res = labelsFromSum(sum, total);
        res.models = total;
        res.ms = Date.now() - t0;
        res.simd = ms.every(function (m) { return m.simd; });
        res.workers = ms.every(function (m) { return m.inWorker; });
        return res;
      });
    });
  };

  return {
    SIZE: SIZE, VALUES: VALUES, Net: Net, Ensemble: Ensemble, normalize: normalize, argmax: argmax, cleanup: cleanup, labelsFromSum: labelsFromSum,
    kernelWasm: kernelWasm, kernelJS: kernelJS, bytesFromBase64: bytesFromBase64
  };
}

var HoasegEngine = hoasegDefine();
if (typeof module !== "undefined" && module.exports) module.exports = HoasegEngine;
