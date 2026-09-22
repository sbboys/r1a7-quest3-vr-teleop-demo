import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import URDFLoader from '/static/vendor/urdf-loader/URDFLoader.js';

const $ = (id) => document.getElementById(id);
const stateEl = $('state');
const logsEl = $('logs');
let robot = null;
let enabled = false;
let repeatTimer = null;
let heldCommand = null;
let heldKind = null;
let cartesianHoldTimer = null;
let cartesianHeartbeatTimer = null;
let cartesianStreaming = false;
let commandQueue = Promise.resolve();
let selectedArm = 'right';
let latestTelemetry = {};
const dexModels = [null, null];
const gripperVisualTarget = [-0.02, -0.02];
const gripperVisualPosition = [-0.02, -0.02];

const gripperCalibration = [
  {open: 4.86, close: -0.08},
  {open: 0.1585, close: -0.05},
];
const dexSliderRange = {open: -0.02, close: 0.0245};
const dexMounts = [
  {position: [0.0415, -0.00013893, -0.02225], rotationZ: -Math.PI / 2},
  {position: [0.041265, -0.00013294, 0.021703], rotationZ: -Math.PI / 2},
];

const jointNames = [
  'left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint',
  'right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint'
];
const jointLabels = ['肩部俯仰','肩部侧摆','肩部旋转','肘部屈伸','腕部翻转','腕部俯仰','腕部旋转'];

function renderJointControls() {
  const offset = selectedArm === 'left' ? 0 : 7;
  $('joint-jog-list').innerHTML = jointLabels.map((name, localIndex) => {
    const index = offset + localIndex;
    return `<div class="joint-row" id="joint-row-${index}">
      <span class="joint-name">${name}</span>
      <span class="joint-angle"><b id="joint-actual-${index}">--</b><small id="joint-goal-${index}">--</small></span>
      <button class="joint-step" data-joint-command="joint:${index}:-" title="${selectedArm === 'left' ? '左' : '右'}臂 ${name}角度减小">−</button>
      <button class="joint-step" data-joint-command="joint:${index}:+" title="${selectedArm === 'left' ? '左' : '右'}臂 ${name}角度增大">+</button>
    </div>`;
  }).join('');
  document.querySelectorAll('[data-joint-command]').forEach((button) => {
    button.disabled = !enabled;
    bindJointJogButton(button);
  });
  updateJointReadouts(latestTelemetry);
}

document.querySelectorAll('[data-arm]').forEach((button) => {
  button.onclick = () => {
    releaseJog();
    selectedArm = button.dataset.arm;
    document.querySelectorAll('[data-arm]').forEach((tab) => {
      const selected = tab === button;
      tab.classList.toggle('selected', selected);
      tab.setAttribute('aria-selected', String(selected));
    });
    renderJointControls();
  };
});

const wrap = $('canvas-wrap');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1318);
scene.fog = new THREE.Fog(0x0d1318, 2.0, 5.0);
const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 20);
camera.position.set(1.4, -1.5, 1.05);
camera.up.set(0, 0, 1);
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.outputColorSpace = THREE.SRGBColorSpace;
wrap.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.55);
controls.enableDamping = true;
controls.minDistance = 0.45;
controls.maxDistance = 4;

scene.add(new THREE.HemisphereLight(0xe9f4ff, 0x26323a, 2.2));
const key = new THREE.DirectionalLight(0xffffff, 2.6);
key.position.set(2, -2, 3);
key.castShadow = true;
scene.add(key);
const grid = new THREE.GridHelper(4, 40, 0x3d4e59, 0x26343d);
grid.rotateX(Math.PI / 2);
scene.add(grid);
scene.add(new THREE.AxesHelper(0.35));

const actualMarker = new THREE.Mesh(new THREE.SphereGeometry(.018, 20, 12), new THREE.MeshStandardMaterial({color: 0x59d48a}));
const commandMarker = new THREE.Mesh(new THREE.SphereGeometry(.014, 20, 12), new THREE.MeshStandardMaterial({color: 0xffb14a, emissive: 0x4a2600}));
scene.add(actualMarker, commandMarker);
actualMarker.visible = commandMarker.visible = false;

function gripperMotorToDexSlider(motorQ, index) {
  const calibration = gripperCalibration[index];
  const openFraction = THREE.MathUtils.clamp(
    (motorQ - calibration.close) / (calibration.open - calibration.close),
    0,
    1,
  );
  return THREE.MathUtils.lerp(dexSliderRange.close, dexSliderRange.open, openFraction);
}

function setDexJointPosition(model, value) {
  model.setJointValue('Joint1_1', value);
  model.setJointValue('Joint2_1', value);
}

function updateGripperVisualTargets(t) {
  const values = [t.gripper_q_actual, t.gripper_q_command, t.gripper_q_goal]
    .find((candidate) => Array.isArray(candidate) && candidate.length >= 2);
  if (!values) return;
  values.slice(0, 2).forEach((value, index) => {
    if (Number.isFinite(value)) gripperVisualTarget[index] = gripperMotorToDexSlider(value, index);
  });
}

function attachDexModel(parentLinkName, index) {
  const dexLoader = new URDFLoader();
  dexLoader.packages = '';
  dexLoader.load('/model/dex1_1/dex1_1.urdf', (model) => {
    const parent = robot?.links?.[parentLinkName];
    if (!parent) {
      appendLog(`Dex1 模型挂载失败: 找不到 ${parentLinkName}`);
      return;
    }
    model.name = index === 0 ? 'left_dex1_visual' : 'right_dex1_visual';
    const mount = dexMounts[index];
    model.position.fromArray(mount.position);
    model.rotation.set(0, 0, mount.rotationZ);
    model.traverse((obj) => {
      if (obj.isMesh) {
        obj.castShadow = true;
        obj.receiveShadow = true;
      }
    });
    parent.add(model);
    dexModels[index] = model;
    gripperVisualPosition[index] = gripperVisualTarget[index];
    setDexJointPosition(model, gripperVisualPosition[index]);
    appendLog(`${index === 0 ? '左' : '右'}侧夹爪三维模型已就绪`);
  }, undefined, (error) => appendLog(`夹爪三维模型加载失败: ${error.message}`));
}

function animateGrippers() {
  dexModels.forEach((model, index) => {
    if (!model) return;
    const next = THREE.MathUtils.lerp(gripperVisualPosition[index], gripperVisualTarget[index], 0.18);
    if (Math.abs(next - gripperVisualPosition[index]) < 1e-6) return;
    gripperVisualPosition[index] = next;
    setDexJointPosition(model, next);
  });
}

const loader = new URDFLoader();
loader.packages = '';
loader.load('/model/A7_Dex1_console.urdf', (model) => {
  robot = model;
  model.rotation.x = 0;
  model.traverse((obj) => { if (obj.isMesh) { obj.castShadow = true; obj.receiveShadow = true; } });
  scene.add(model);
  attachDexModel('left_wrist_yaw_link', 0);
  attachDexModel('right_wrist_yaw_link', 1);
}, undefined, (error) => appendLog(`模型加载失败: ${error.message}`));

function resize() {
  const {clientWidth:w, clientHeight:h} = wrap;
  camera.aspect = w / Math.max(1, h);
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}
new ResizeObserver(resize).observe(wrap);
function animate() { requestAnimationFrame(animate); controls.update(); animateGrippers(); renderer.render(scene, camera); }
resize(); animate();

function resetView() { camera.position.set(1.4, -1.5, 1.05); controls.target.set(0, 0, .55); controls.update(); }
$('reset-view').onclick = resetView;
$('toggle-grid').onclick = () => { grid.visible = !grid.visible; };

function appendLog(line) {
  logsEl.textContent += `${line}\n`;
  const lines = logsEl.textContent.split('\n');
  if (lines.length > 180) logsEl.textContent = lines.slice(-180).join('\n');
  logsEl.scrollTop = logsEl.scrollHeight;
}
$('clear-log').onclick = () => { logsEl.textContent = ''; };

function setPhase(phase) {
  const labels = {stopped:'系统待机', starting:'正在连接', enabling:'正在授权', waiting_enable:'等待授权', enabled:'控制已连接', stopping:'正在断开', fault:'系统异常'};
  stateEl.textContent = labels[phase] || phase;
  stateEl.dataset.state = phase;
  enabled = phase === 'enabled';
  $('start').disabled = !['stopped', 'fault'].includes(phase);
  $('enable').disabled = phase !== 'waiting_enable';
  $('stop').disabled = phase === 'stopped';
  document.querySelectorAll('[data-command], [data-joint-command], [data-gripper-command]').forEach((b) => { b.disabled = !enabled; });
}

async function api(path, body={}) {
  const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

$('start').onclick = async () => {
  try {
    await api('/api/start', {interface:$('interface').value, host_ip:$('host-ip').value, xy_step_mm:Number($('xy-step').value), z_step_mm:Number($('z-step').value), joint_step_deg:Number($('joint-step').value), gripper_speed:Number($('gripper-speed').value), cartesian_speed_mm_s:Number($('cartesian-speed').value), cartesian_accel_mm_s2:Number($('cartesian-accel').value), right_arm_gravity_ff_scale:Number($('gravity-ff-scale').value)});
    setPhase('starting');
  } catch (e) { appendLog(`连接失败: ${e.message}`); }
};
$('enable').onclick = async () => {
  try {
    setPhase('enabling');
    appendLog('[控制] 已发送授权命令，等待底层控制初始化');
    await api('/api/command', {command:'ENABLE'});
  } catch(e) {
    appendLog(`授权控制失败: ${e.message}`);
    setPhase('waiting_enable');
  }
};
$('stop').onclick = async () => { releaseJog(); try { await api('/api/stop'); setPhase('stopped'); } catch(e) { appendLog(e.message); } };

function sendCommand(command) {
  if (!enabled) return Promise.resolve();
  commandQueue = commandQueue
    .then(() => api('/api/command', {command}))
    .catch((e) => {
      appendLog(`命令失败: ${e.message}`);
      releaseJog(false);
    });
  return commandQueue;
}
function pressJointJog(button) {
  if (!enabled || heldCommand) return;
  heldCommand = button.dataset.jointCommand;
  heldKind = 'joint';
  button.classList.add('active');
  sendCommand(heldCommand);
  repeatTimer = setInterval(() => sendCommand(heldCommand), Math.max(120, Number($('repeat-ms').value)));
}
function pressCartesianJog(button) {
  if (!enabled || heldCommand) return;
  heldCommand = button.dataset.command;
  heldKind = 'cartesian';
  button.classList.add('active');
  cartesianHoldTimer = setTimeout(() => {
    cartesianHoldTimer = null;
    if (heldKind !== 'cartesian' || !heldCommand) return;
    cartesianStreaming = true;
    sendCommand(`jog:start:${heldCommand}`);
    cartesianHeartbeatTimer = setInterval(() => {
      if (cartesianStreaming && heldCommand) sendCommand(`jog:keepalive:${heldCommand}`);
    }, 150);
  }, 180);
}
function releaseJog(sendClick=true) {
  const releasedCommand = heldCommand;
  const releasedKind = heldKind;
  const wasStreaming = cartesianStreaming;
  if (cartesianHoldTimer) clearTimeout(cartesianHoldTimer);
  if (cartesianHeartbeatTimer) clearInterval(cartesianHeartbeatTimer);
  cartesianHoldTimer = null;
  cartesianHeartbeatTimer = null;
  cartesianStreaming = false;
  heldCommand = null;
  heldKind = null;
  if (repeatTimer) clearInterval(repeatTimer);
  repeatTimer = null;
  document.querySelectorAll('[data-command], [data-joint-command]').forEach((b) => b.classList.remove('active'));
  if (releasedKind === 'cartesian') {
    if (wasStreaming) sendCommand('jog:stop');
    else if (sendClick && releasedCommand) sendCommand(releasedCommand);
  }
}
function bindJointJogButton(button) {
  button.addEventListener('pointerdown', (e) => { e.preventDefault(); button.setPointerCapture(e.pointerId); pressJointJog(button); });
  button.addEventListener('pointerup', releaseJog);
  button.addEventListener('pointercancel', () => releaseJog(false));
}
function bindCartesianJogButton(button) {
  button.addEventListener('pointerdown', (e) => { e.preventDefault(); button.setPointerCapture(e.pointerId); pressCartesianJog(button); });
  button.addEventListener('pointerup', () => releaseJog(true));
  button.addEventListener('pointercancel', () => releaseJog(false));
}
document.querySelectorAll('.jog [data-command]').forEach(bindCartesianJogButton);
document.querySelectorAll('.utility-actions [data-command]').forEach((button) => {
  if (button.dataset.command === 'i') bindCartesianJogButton(button);
  else button.onclick = () => sendCommand(button.dataset.command);
});
document.querySelectorAll('[data-gripper-command]').forEach((button) => {
  button.onclick = () => sendCommand(button.dataset.gripperCommand);
});
window.addEventListener('blur', () => releaseJog(false));
window.addEventListener('pointerup', () => releaseJog(true));

function updateJointReadouts(t) {
  const q = t.arm_q_actual || [];
  const goal = t.arm_q_goal || [];
  for (let i = 0; i < 14; i += 1) {
    const actualEl = $(`joint-actual-${i}`);
    const goalEl = $(`joint-goal-${i}`);
    if (actualEl && Number.isFinite(q[i])) actualEl.textContent = `${(q[i] * 180 / Math.PI).toFixed(1)}°`;
    if (goalEl && Number.isFinite(goal[i])) goalEl.textContent = `${(goal[i] * 180 / Math.PI).toFixed(1)}°`;
    const row = $(`joint-row-${i}`);
    if (row) row.classList.toggle('active-joint', t.control_mode === 'joint' && t.active_joint_index === i);
  }
}

function updateGripperReadouts(t) {
  const actual = t.gripper_q_actual || [];
  const goal = t.gripper_q_goal || [];
  const contact = t.gripper_contact_hold || [];
  for (let i = 0; i < 2; i += 1) {
    const actualEl = $(`gripper-actual-${i}`);
    const goalEl = $(`gripper-goal-${i}`);
    const stateEl = $(`gripper-state-${i}`);
    if (actualEl) actualEl.textContent = Number.isFinite(actual[i]) ? actual[i].toFixed(4) : '--';
    if (goalEl) goalEl.textContent = Number.isFinite(goal[i]) ? goal[i].toFixed(4) : '--';
    if (stateEl) {
      if (contact[i]) stateEl.textContent = '接触保持中';
      else if (Number.isFinite(actual[i]) && Number.isFinite(goal[i]) && Math.abs(actual[i] - goal[i]) > 0.02) stateEl.textContent = '运动中';
      else if (Number.isFinite(actual[i])) stateEl.textContent = '位置保持中';
      else stateEl.textContent = '--';
    }
  }
}

function updateTelemetry(t) {
  latestTelemetry = t || {};
  const q = t.arm_q_actual || [];
  if (robot && q.length === 14) jointNames.forEach((name, i) => robot.setJointValue(name, q[i]));
  updateGripperVisualTargets(t);
  updateJointReadouts(t);
  updateGripperReadouts(t);
  const directionLabels = {w:'前 +X', s:'后 -X', a:'左 +Y', d:'右 -Y', u:'上 +Z', j:'下 -Z', i:'前上 +X+Z'};
  const motionLabels = {continuous_jog:'连续点动', decelerating:'平滑减速', holding:'位置保持'};
  const baseMode = t.control_mode === 'joint' ? '关节控制' : t.control_mode === 'cartesian' ? '末端控制' : '--';
  $('control-mode').textContent = t.jog_direction
    ? `${baseMode} · ${directionLabels[t.jog_direction] || t.jog_direction}`
    : `${baseMode}${t.motion_state ? ` · ${motionLabels[t.motion_state] || t.motion_state}` : ''}`;
  const fmt = (v) => Array.isArray(v) ? `[${v.map(x => x.toFixed(1)).join(', ')}] mm` : '--';
  $('actual-xyz').textContent = fmt(t.actual_xyz_mm);
  $('command-xyz').textContent = fmt(t.command_xyz_mm);
  $('lag').textContent = Number.isFinite(t.lag_mm) ? `${t.lag_mm.toFixed(1)} mm` : '--';
  if ($('orientation-error')) $('orientation-error').textContent = Number.isFinite(t.orientation_error_deg) ? `${t.orientation_error_deg.toFixed(2)}°` : '--';
  if ($('command-speed')) $('command-speed').textContent = Number.isFinite(t.command_velocity_mm_s) ? `${t.command_velocity_mm_s.toFixed(1)} mm/s` : '--';
  if ($('jacobian-condition')) {
    const condition = t.right_jacobian_condition;
    $('jacobian-condition').textContent = Number.isFinite(condition) ? condition.toFixed(0) : '--';
    $('jacobian-condition').dataset.level = Number.isFinite(condition) && condition >= 500 ? 'critical' : Number.isFinite(condition) && condition >= 250 ? 'warning' : 'normal';
  }
  if (Array.isArray(t.actual_xyz_mm)) { actualMarker.position.fromArray(t.actual_xyz_mm.map(x=>x/1000)); actualMarker.visible=true; }
  if (Array.isArray(t.command_xyz_mm)) { commandMarker.position.fromArray(t.command_xyz_mm.map(x=>x/1000)); commandMarker.visible=true; }
}

const wsProtocol = location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WebSocket(`${wsProtocol}://${location.host}/ws`);
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.type === 'status') { setPhase(message.data.phase); (message.data.logs || []).forEach(appendLog); if(message.data.telemetry) updateTelemetry(message.data.telemetry); }
  if (message.type === 'log') { setPhase(message.phase); appendLog(message.data); }
  if (message.type === 'telemetry') updateTelemetry(message.data);
};
ws.onclose = () => { setPhase('fault'); appendLog('控制服务连接已断开'); };
renderJointControls();
setPhase('stopped');
