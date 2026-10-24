// worldlines motion workbench: four pipeline stages (来源 → 动作 → 身体 → 导出) around one SMPL-X stage.
// Stage rendering, layout and job flow adapted from prompt2rig (Apache-2.0, Copyright 2026 blendi-remade).
import { createStage } from './stage.js';
import { SmplxMesh, SkeletonLines } from './smplx_mesh.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const PRESET_PROMPTS = [
  'walks in, stops, does a deep squat and stands back up', 'gets startled, jumps and lands', 'does an energetic hip hop dance with wide arm movements',
  'throws boxing jabs and hooks toward the camera', 'does jumping jacks', 'waves hello with both arms, then takes a bow',
  'celebrates a victory, jumping and pumping both fists in the air', 'does a karate kata with straight punches and front kicks',
];
const DEFAULT_PHYSIQUE = { betas: Array(10).fill(0), muscular: 0, chest: 0, waist: 0, bulge: 0, bulge_size: 1 };

// ---- small helpers ---------------------------------------------------------------------------
function log(msg, cls = '') { const el = $('#log'); const d = document.createElement('div'); if (cls) d.className = cls; d.textContent = `${new Date().toLocaleTimeString([], { hour12: false })}  ${msg}`; el.appendChild(d); el.scrollTop = el.scrollHeight; if (cls === 'err') $('.logbox').open = true; }
function pill(id, text, state) { const p = $('#pill-' + id); p.textContent = text; p.className = 'pill ' + (state || ''); }
function banner(text) { const b = $('#banner'); b.textContent = text || ''; b.classList.toggle('on', !!text); }
function fmt(v) { return typeof v === 'number' ? (Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(1) : v.toFixed(3)) : String(v ?? '—'); }
async function api(path, body, method) {
  const res = await fetch(path, { method: method || (body ? 'POST' : 'GET'), headers: body ? { 'Content-Type': 'application/json' } : {}, body: body && JSON.stringify(body) });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(json.error || `${path} → ${res.status}`);
  return json;
}
async function waitJob(id, onProgress) {
  for (;;) {
    const j = await api('/api/job/' + id);
    if (j.status === 'done') return j.result;
    if (j.status === 'error') throw new Error(j.error);
    onProgress?.(j.progress);
    await new Promise((r) => setTimeout(r, 700));
  }
}

// ---- pipeline state --------------------------------------------------------------------------
let raw = null, natural = null, showNatural = true, clipPath = null, savedPath = null, presets = {};
let physique = { ...DEFAULT_PHYSIQUE };
const cur = () => (showNatural && natural) || raw;
let frame = 0, playing = false, lastT = 0, acc = 0;

function setStage(name) {
  $$('.step').forEach((b) => b.classList.toggle('on', b.dataset.stage === name));
  $$('.panel').forEach((p) => p.classList.toggle('on', p.dataset.stage === name));
  try { localStorage.setItem('wl.stage', name); } catch {}
}
$$('.step').forEach((b) => (b.onclick = () => setStage(b.dataset.stage)));

function physiqueLabel() {
  const hit = Object.entries(presets).find(([, p]) => ['muscular', 'chest', 'waist', 'bulge', 'bulge_size'].every((k) => Math.abs((p[k] ?? 0) - (physique[k] ?? 0)) < 1e-6) && (p.betas || []).every((b, i) => Math.abs(b - (physique.betas?.[i] ?? 0)) < 1e-3));
  if (hit) return hit[0];
  const parts = ['muscular', 'chest', 'waist', 'bulge'].filter((k) => physique[k] > 0).map((k) => `${k} ${physique[k].toFixed(2)}`);
  return parts.length ? parts.join(' · ') : 'custom betas';
}
function updateStatus() {
  const c = cur();
  const done = (stage, on) => $(`.step[data-stage=${stage}]`).classList.toggle('done', !!on);
  $('#st-clip').textContent = raw ? `${clipPath ? clipPath.split('/').pop() : '(未保存)'} · ${raw.source || '?'} · ${raw.n}f @ ${raw.fps}` : '—';
  $('#st-motion').textContent = raw ? (natural ? (showNatural ? '已自然化（显示 natural）' : '已自然化（显示 raw）') : 'raw，未自然化') : '—';
  $('#st-motion').className = natural ? 'good' : '';
  $('#st-body').textContent = physiqueLabel();
  $('#st-body').className = physiqueLabel() !== 'neutral' ? 'good' : '';
  $('#st-saved').textContent = savedPath || (clipPath ? '来自 ' + clipPath : '—');
  done('source', !!raw); done('motion', !!natural); done('body', physiqueLabel() !== 'neutral'); done('export', !!savedPath);
  $('#empty').style.display = raw ? 'none' : 'flex';
  if (c) $('#save-name').placeholder = (clipPath ? clipPath.split('/').pop().replace(/\.raw\.smplx\.npz$|\.smplx\.npz$|\.npz$/, '') : 'take') + (showNatural && natural ? '_natural' : '');
}

// ---- stage + mesh ----------------------------------------------------------------------------
const stage = createStage($('#stage'));
stage.setInset(0);
let body = null, skel = null, modelPayload = null;

async function loadModel() {
  pill('model', 'loading smpl-x…', 'busy');
  modelPayload = await api('/api/model?physique=' + encodeURIComponent(JSON.stringify(physique)));
  if (body) { stage.scene.remove(body.mesh); stage.removeContactShadow(body.mesh); }
  body = new SmplxMesh(modelPayload);
  stage.scene.add(body.mesh); stage.contactShadow(body.mesh);
  if (!skel) { skel = new SkeletonLines(modelPayload.parents); stage.scene.add(skel.lines); skel.lines.visible = false; }
  pill('model', `smpl-x ${modelPayload.gender} · ${modelPayload.vertices.length / 3} verts`, 'ok');
  log(`model loaded (${modelPayload.gender}, ${modelPayload.influences} influences/vertex)`);
  renderFrame();
}

let physiqueTimer = null;
function physiqueChanged() {
  clearTimeout(physiqueTimer);
  physiqueTimer = setTimeout(async () => {
    try {
      const r = await api('/api/model?only=vertices&physique=' + encodeURIComponent(JSON.stringify(physique)));
      body.setRest(r.vertices, r.joints); physique = { ...physique, ...r.profile };
      renderFrame(); updateStatus();
    } catch (e) { log('physique failed: ' + e.message, 'err'); }
  }, 120);
}

function renderFrame() {
  const c = cur();
  if (!body) return;
  if (!c) { body.pose(new Float32Array(3), new Float32Array(63), [0, -modelPayload.sole_offset, 0]); if (skel) skel.update(body.joints); stage.placeContactShadow(body.mesh, body.joint(0), 0, 1.7); return; }
  const f = Math.min(frame, c.n - 1), o3 = f * 3, o63 = f * 63;
  const go = c.global_orient.slice(o3, o3 + 3), bp = c.body_pose.slice(o63, o63 + 63), tr = c.transl.slice(o3, o3 + 3);
  const lh = c.left_hand_pose ? c.left_hand_pose.slice(f * 45, f * 45 + 45) : null, rh = c.right_hand_pose ? c.right_hand_pose.slice(f * 45, f * 45 + 45) : null;
  body.pose(go, bp, tr, lh, rh, c.jaw_pose ? c.jaw_pose.slice(o3, o3 + 3) : null);
  if (skel) skel.update(body.joints);
  stage.placeContactShadow(body.mesh, body.joint(0), 0, 1.75);
  $('#hud-src').textContent = `${c.source || 'clip'} · ${showNatural && natural ? 'natural' : 'raw'}`;
  $('#hud-time').textContent = `frame ${f + 1}/${c.n}  ${(f / c.fps).toFixed(2)}s`;
  $('#time').textContent = `${f + 1} / ${c.n}  ·  ${(f / c.fps).toFixed(2)} s  ·  ${c.fps} fps`;
  $('#scrub').value = String(f);
  const v = $('#video');
  if (c.video && v.readyState >= 1 && !playing) v.currentTime = Math.min(v.duration || 0, f / c.fps);
}

function tick(t) {
  requestAnimationFrame(tick);
  const c = cur();
  if (playing && c) {
    const dt = Math.min(0.1, (t - lastT) / 1000) * Number($('#speed').value);
    acc += dt * c.fps;
    if (acc >= 1) { frame = (frame + Math.floor(acc)) % c.n; acc -= Math.floor(acc); renderFrame(); }
  }
  lastT = t;
  stage.update(body && $('#v-follow').classList.contains('on') ? body.joint(0) : null);
}
requestAnimationFrame(tick);

function togglePlay(on) {
  playing = on ?? !playing; $('#play').textContent = playing ? '❚❚' : '▶';
  const v = $('#video'); if (playing) { v.playbackRate = Number($('#speed').value); v.currentTime = frame / (cur()?.fps || 30); v.play().catch(() => {}); } else v.pause();
}
$('#play').onclick = () => togglePlay();
$('#scrub').oninput = (e) => { playing && togglePlay(false); frame = Number(e.target.value); renderFrame(); };
$('#speed').onchange = () => { $('#video').playbackRate = Number($('#speed').value); };
window.addEventListener('keydown', (e) => {
  if (['TEXTAREA', 'INPUT', 'SELECT'].includes(e.target.tagName)) return;
  if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  if (e.key === 'ArrowRight') { frame = Math.min(frame + 1, (cur()?.n || 1) - 1); renderFrame(); }
  if (e.key === 'ArrowLeft') { frame = Math.max(0, frame - 1); renderFrame(); }
  if (e.key >= '1' && e.key <= '4') setStage(['source', 'motion', 'body', 'export'][Number(e.key) - 1]);
});

// ---- clip handling ---------------------------------------------------------------------------
function setClip(rawClip, natClip, path, { advance = false } = {}) {
  raw = rawClip; natural = natClip || null; clipPath = path || null; savedPath = null; frame = 0;
  $('#scrub').max = String(Math.max(0, raw.n - 1));
  showMetrics();
  if (raw.video) { $('#video').src = raw.video; $('#vidwrap').style.display = 'block'; } else { $('#vidwrap').style.display = 'none'; }
  $('#btn-ab').textContent = '显示: ' + (showNatural && natural ? 'natural' : 'raw');
  renderFrame(); renderClipList(); updateStatus();
  if (advance) setStage('motion');
  if (!playing) togglePlay(true);
}

function showMetrics() {
  const el = $('#metrics');
  const keys = ['frames', 'fps', 'angular_jerk', 'mean_angular_speed', 'foot_skate_m_per_s', 'contact_skate_m_per_s', 'contact_ratio', 'floor_penetration_m'];
  const a = raw?.metrics || {}, b = natural?.metrics || {};
  el.innerHTML = `<span></span><span class="h">raw</span><span class="h">natural</span>` + keys.map((k) => `<span>${k}</span><b>${fmt(a[k])}</b><b>${fmt(b[k])}</b>`).join('');
}

async function renderClipList() {
  const el = $('#clips');
  let list = [];
  try { list = await api('/api/clips'); } catch (e) { log('clip list failed: ' + e.message, 'err'); }
  el.innerHTML = '';
  for (const c of list) {
    const d = document.createElement('div'); d.className = 'clipcard' + (c.path === clipPath ? ' on' : ''); d.title = c.path;
    d.innerHTML = `<span>${c.name}</span><span>${c.frames}f @ ${c.fps}</span><small>${c.path.split('/').slice(0, -1).join('/')} · ${c.source || '?'}</small>`;
    d.onclick = () => loadClip(c.path);
    el.appendChild(d);
  }
}
$('#btn-refresh').onclick = renderClipList;

async function loadClip(path) {
  banner('loading ' + path);
  try {
    const c = await api('/api/clip?path=' + encodeURIComponent(path));
    physique = { ...DEFAULT_PHYSIQUE, ...(c.physique || {}) }; syncPhysiqueUI(); physiqueChanged();
    setClip(c, null, path); log(`loaded ${path} (${c.n} frames, ${c.source})`, 'ok');
  } catch (e) { log('load failed: ' + e.message, 'err'); }
  banner('');
}

// ---- 1 source: prompt ------------------------------------------------------------------------
$('#duration').oninput = (e) => ($('#duration-v').textContent = e.target.value + 's');
for (const p of PRESET_PROMPTS) { const c = document.createElement('span'); c.className = 'chip'; c.textContent = p; c.onclick = () => ($('#prompt').value = p); $('#chips').appendChild(c); }
$('#go').onclick = async () => {
  const prompt = $('#prompt').value.trim(); if (!prompt) return;
  $('#go').disabled = true; banner('generating video…');
  try {
    const { job } = await api('/api/prompt_to_motion', { prompt, video_model: $('#videoModel').value, duration: Number($('#duration').value), orientation: $('#orientation').value, backend: $('#backend').value, natural: natConfig() });
    const r = await waitJob(job, (p) => banner(p || 'working…'));
    setClip(r.clip, null, r.path, { advance: true }); log(`prompt → motion done (${r.clip.n} frames, ${Object.entries(r.timings).map(([k, v]) => `${k} ${v.toFixed(1)}s`).join(', ')})`, 'hq');
  } catch (e) { log('prompt→motion failed: ' + e.message, 'err'); }
  $('#go').disabled = false; banner('');
};
$('#prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $('#go').click(); } });

// ---- 1 source: video -------------------------------------------------------------------------
$('#btn-vid').onclick = () => $('#file-vid').click();
$('#file-vid').onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  e.target.value = '';
  banner('uploading ' + f.name);
  try {
    const up = await fetch('/api/upload?name=' + encodeURIComponent(f.name), { method: 'POST', body: f }).then((r) => r.json());
    log(`uploaded ${up.name}`);
    const { job } = await api('/api/video_to_motion', { name: up.name, backend: $('#backend').value, natural: natConfig() });
    pill('v2m', 'video→motion running', 'busy');
    const r = await waitJob(job, (p) => banner(p || 'video → motion…'));
    setClip(r.raw, r.clip, r.path, { advance: true }); log(`motion from video: ${r.clip.n} frames via ${r.clip.source}, saved ${r.path}`, 'hq');
  } catch (err) { log('video→motion failed: ' + err.message, 'err'); }
  banner(''); checkHealth();
};

// ---- 2 motion: naturalize --------------------------------------------------------------------
const natKeys = ['smooth_sigma', 'root_sigma', 'velocity_adaptive', 'joint_limits', 'ground', 'foot_lock', 'settle_ends', 'breathing'];
function natConfig() { const c = {}; for (const k of natKeys) { const el = $('#' + k); c[k] = el.type === 'checkbox' ? el.checked : Number(el.value); } return c; }
for (const k of ['smooth_sigma', 'root_sigma', 'breathing']) { const el = $('#' + k); const show = () => ($('#' + k + '-v').textContent = el.value); el.oninput = show; show(); }
$('#btn-nat').onclick = async () => {
  if (!raw) { setStage('source'); return log('先在第 1 步载入一条片段', 'err'); }
  banner('naturalizing…');
  try {
    const r = await api('/api/naturalize', { clip: raw, config: natConfig() });
    natural = r; showNatural = true; const keep = frame; setClip(raw, natural, clipPath); frame = keep; renderFrame();
    log(`naturalized: jerk ${fmt(raw.metrics?.angular_jerk)} → ${fmt(natural.metrics?.angular_jerk)}, contact skate ${fmt(raw.metrics?.contact_skate_m_per_s)} → ${fmt(natural.metrics?.contact_skate_m_per_s)}`, 'hq');
  } catch (e) { log('naturalize failed: ' + e.message, 'err'); }
  banner('');
};
$('#btn-ab').onclick = () => { showNatural = !showNatural; $('#btn-ab').textContent = '显示: ' + (showNatural && natural ? 'natural' : 'raw'); renderFrame(); updateStatus(); };
$('#btn-reset').onclick = () => { natural = null; showNatural = true; const keep = frame; setClip(raw, null, clipPath); frame = keep; renderFrame(); };

// ---- 3 body: physique ------------------------------------------------------------------------
const physKeys = ['muscular', 'chest', 'waist', 'bulge', 'bulge_size'];
function syncPhysiqueUI() {
  for (const k of physKeys) { $('#' + k).value = physique[k] ?? 0; $('#' + k + '-v').textContent = Number($('#' + k).value).toFixed(2); }
  $('#betas').value = (physique.betas || Array(10).fill(0)).map((b) => +Number(b).toFixed(2)).join(',');
  const name = physiqueLabel(); if (presets[name]) $('#preset').value = name;
}
for (const k of physKeys) { $('#' + k).oninput = (e) => { physique[k] = Number(e.target.value); $('#' + k + '-v').textContent = Number(e.target.value).toFixed(2); physiqueChanged(); }; }
$('#betas').onchange = (e) => { const b = e.target.value.split(',').map(Number); if (b.length === 10 && b.every(Number.isFinite)) { physique.betas = b; physiqueChanged(); } else log('betas need 10 numbers', 'err'); };
api('/api/presets').then((p) => {
  presets = p; const s = $('#preset');
  for (const k of Object.keys(p)) { const o = document.createElement('option'); o.value = k; o.textContent = k; s.appendChild(o); }
  s.onchange = () => { physique = { ...DEFAULT_PHYSIQUE, ...p[s.value] }; syncPhysiqueUI(); physiqueChanged(); };
  syncPhysiqueUI(); updateStatus();
});
syncPhysiqueUI();

// ---- 4 export --------------------------------------------------------------------------------
$('#btn-save').onclick = async () => {
  const c = cur(); if (!c) { setStage('source'); return log('先载入一条片段', 'err'); }
  const name = $('#save-name').value.trim() || $('#save-name').placeholder;
  try { const r = await api('/api/save', { clip: c, name, physique }); log('saved ' + r.path, 'ok'); clipPath = r.path; savedPath = r.path; renderClipList(); updateStatus(); }
  catch (e) { log('save failed: ' + e.message, 'err'); }
};
$('#btn-blender').onclick = () => {
  const p = savedPath || clipPath;
  if (!p) return log('先保存片段（Blender 导出读取已保存的文件）', 'err');
  if (natural && !savedPath) log('提示：导出的是磁盘上的文件；若要导出自然化后的版本，先保存', 'hq');
  window.open('/api/export/blender?path=' + encodeURIComponent(p));
};
$('#btn-json').onclick = () => { const c = cur(); if (!c) return; const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify({ ...c, physique })], { type: 'application/json' })); a.download = ($('#save-name').value.trim() || $('#save-name').placeholder || 'clip') + '.json'; a.click(); };

// ---- view toggles ----------------------------------------------------------------------------
const toggle = (id, fn) => { const b = $(id); b.onclick = () => { b.classList.toggle('on'); fn(b.classList.contains('on')); }; };
toggle('#v-skel', (on) => { if (skel) skel.lines.visible = on; });
toggle('#v-cine', (on) => stage.setCinematic(on));
toggle('#v-follow', () => {});

// ---- boot ------------------------------------------------------------------------------------
async function checkHealth() {
  try {
    const h = await api('/api/health');
    pill('v2m', h.gvhmr ? 'gvhmr worker online' : h.mediapipe ? 'mediapipe (cpu) ready' : 'no video→motion backend', h.gvhmr ? 'ok' : h.mediapipe ? '' : 'bad');
    pill('fal', h.fal ? 'fal key set' : 'no FAL_KEY', h.fal ? 'ok' : 'bad');
    $('#blk-prompt').classList.toggle('off', !h.fal);
    $('#note-prompt').textContent = h.fal ? '用 fal.ai 先生成一段演员视频，再从视频里提动作。' : '未设置 FAL_KEY，文字入口不可用。把 key 写进仓库根目录的 .env 后重启工作台；或者直接走视频入口。';
    if (h.smplx_error) { pill('model', 'smpl-x missing', 'bad'); log(h.smplx_error, 'err'); }
    return h;
  } catch (e) { log('health failed: ' + e.message, 'err'); }
}
window.wl = { get raw() { return raw; }, get natural() { return natural; }, get body() { return body; }, stage, setFrame(f) { frame = f; renderFrame(); }, setStage };
(async () => {
  const h = await checkHealth();
  if (h?.smplx) await loadModel().catch((e) => { log('model failed: ' + e.message, 'err'); pill('model', 'model failed', 'bad'); });
  await renderClipList();
  updateStatus();
  try { const s = localStorage.getItem('wl.stage'); if (s && !raw) setStage('source'); } catch {}
  log('ready: 1 来源 → 2 动作 → 3 身体 → 4 导出');
})();
