"""
C. elegans LIF neural simulation + Ursina 3D free camera
Environment: Windows 11 + Python 3.12
Install: pip install ursina numpy

Controls:
    WASD        Move
    Mouse       Look around
    Space / Q   Up / Down
    Shift       Boost
    ESC         Lock / unlock mouse
    R           Reset camera
"""

import math
import numpy as np
from ursina import *

# ====================================================
# ========== 1. 神经参数 =============================
# ====================================================
N_NEURONS = 302
N_SEGMENTS = 17

V_rest   = -65.0
V_thresh = -50.0
V_reset  = -70.0
V_min    = -100.0
V_max    =   50.0
tau_m    = 20.0
R_m      = 10.0
dt_sim   = 1.0

# ====================================================
# ========== 2. 合成连接组 ===========================
# ====================================================
np.random.seed(42)

n_exc = int(N_NEURONS * 0.8)
is_excitatory = np.zeros(N_NEURONS, dtype=bool)
is_excitatory[:n_exc] = True

connection_prob = 0.05
mask = np.random.rand(N_NEURONS, N_NEURONS) < connection_prob
np.fill_diagonal(mask, False)

W = np.zeros((N_NEURONS, N_NEURONS))
W[mask] = np.random.uniform(0.5, 2.0, mask.sum())
W[:, ~is_excitatory] *= -1
W = W * (5.0 / np.max(np.abs(W)))

print(f"[connectome] generated {mask.sum()} synapses")

sensory_left  = list(range(20, 30))
sensory_right = list(range(30, 40))
inter_left    = list(range(40, 50))
inter_right   = list(range(50, 60))

motor_start     = N_NEURONS - 40
forward_neurons = list(range(motor_start, motor_start + 15))
left_neurons    = list(range(motor_start + 15, motor_start + 27))
right_neurons   = list(range(motor_start + 27, motor_start + 40))

def add_connections(src_list, dst_list, weight):
    for s in src_list:
        for d in dst_list:
            W[s, d] += weight

add_connections(sensory_left,  inter_left,  1.2)
add_connections(inter_left,    left_neurons, 1.0)
add_connections(sensory_right, inter_right, 1.2)
add_connections(inter_right,   right_neurons, 1.0)

for s in inter_left:
    for d in right_neurons:
        W[s, d] -= 0.8
for s in inter_right:
    for d in left_neurons:
        W[s, d] -= 0.8

W = W * (5.0 / max(1e-9, np.max(np.abs(W))))

# ====================================================
# ========== 3. 神经状态 =============================
# ====================================================
V = np.full(N_NEURONS, V_rest)
last_spikes = np.zeros(N_NEURONS)
total_steps = 0

I_ext = np.zeros(N_NEURONS)
I_ext[0:20] = 8.0

# ====================================================
# ========== 5. 世界参数 =============================
# ====================================================
WORLD_RADIUS = 9.0
BASE_SPEED   = 2.2
SPACING      = 0.20

SPRING_K     = 12.0
DAMPING_C    = 10.0

# ---- 身体行波（横向背腹波）----
WAVE_AMP_BODY   = 0.045
WAVE_FREQ_BODY  = 0.55
WAVE_BASE_SPEED = 6.0
WAVE_SPEED_GAIN = 0.6
WAVE_DECAY_BODY = 0.92

# ---- 头部摆动 ----
HEAD_SWING_AMP   = 0.10
HEAD_SWING_FREQ  = 1.1

# ---- 转角惯性 ----
TURN_TAU   = 0.18
turn_smoothed = 0.0

# ---- 趋化 ----
FOOD_SIGMA   = 0.8
PROBE        = 0.3

DT_MAX       = 0.033

# ====================================================
# ========== 6. Ursina 场景 ==========================
# ====================================================
app = Ursina(title='C. elegans Virtual Environment')

DirectionalLight(y=8, z=-5, rotation=(50, -30, 0))
AmbientLight(color=color.rgba(120, 120, 140, 255))

Entity(
    model='circle',
    scale=WORLD_RADIUS * 2,
    color=color.rgb(0.20, 0.16, 0.12),
    position=(0, 0, 0),
    rotation_x=90,
)
for r, col in [(WORLD_RADIUS, color.rgb(0.35, 0.30, 0.25)),
               (WORLD_RADIUS + 0.3, color.rgb(0.25, 0.22, 0.18))]:
    Entity(model='circle', scale=r * 2, color=col,
           position=(0, -0.01, 0), rotation_x=90)

for i in range(24):
    a = i * math.pi * 2 / 24
    Entity(model='cube', scale=(0.15, 0.6, 0.15),
           position=(WORLD_RADIUS * math.cos(a), 0.3, WORLD_RADIUS * math.sin(a)),
           color=color.rgb(0.5, 0.45, 0.35))

foods = []
def spawn_food():
    angle = np.random.rand() * math.pi * 2
    radius = np.random.rand() * (WORLD_RADIUS - 1.5)
    f = Entity(model='sphere', color=color.rgb(0.35, 0.95, 0.35),
               scale=0.22,
               position=(radius * math.cos(angle), 0.15, radius * math.sin(angle)))
    Entity(parent=f, model='sphere',
           color=color.rgba(0.3, 1.0, 0.3, 80), scale=1.8)
    foods.append(f)

for _ in range(20):
    spawn_food()

score = 0

# ====================================================
# ========== 7. 线虫身体 =============================
# ====================================================
base_sizes = [0.15 * (1.0 - 0.55 * (i / (N_SEGMENTS - 1))) for i in range(N_SEGMENTS)]
worm_segments = [Entity(model='sphere',
                        color=color.rgb(0.95, 0.86, 0.72),
                        scale=base_sizes[i]) for i in range(N_SEGMENTS)]

eye_l = Entity(model='sphere', color=color.black, scale=0.035)
eye_r = Entity(model='sphere', color=color.black, scale=0.035)

# ====================================================
# ========== 8. 位置状态 =============================
# ====================================================
head_pos = Vec3(0, 0.1, 0)
head_angle = 0.0
seg_positions  = [Vec3(-i * SPACING, 0.1, 0) for i in range(N_SEGMENTS)]
seg_velocities = [Vec3(0, 0, 0) for _ in range(N_SEGMENTS)]

first_frame = True
wave_phase  = 0.0

# ====================================================
# ========== 9. 自由相机 =============================
# ====================================================
CAM_DEFAULT_POS   = Vec3(0, 10, -14)
CAM_DEFAULT_YAW   = 0
CAM_DEFAULT_PITCH = 25

cam_yaw, cam_pitch = CAM_DEFAULT_YAW, CAM_DEFAULT_PITCH
CAM_SENSITIVITY = 40
CAM_SPEED       = 8
CAM_BOOST       = 3

camera.position = CAM_DEFAULT_POS
camera.rotation = (cam_pitch, cam_yaw, 0)
mouse.locked = True

# ====================================================
# ========== 10. UI ==================================
# ====================================================
info_text = Text(text='', position=(-0.85, 0.45), scale=1.0,
                 color=color.white, background=True)
help_text = Text(
    text='WASD move | Mouse look | Space/Q up-down | Shift boost | ESC unlock mouse | R reset camera',
    position=(-0.85, -0.47), scale=0.85,
    color=color.rgb(0.85, 0.85, 0.95), background=True)

# ====================================================
# ========== 11. 相机输入处理 ========================
# ====================================================
def update_camera():
    global cam_yaw, cam_pitch
    dt = min(time.dt, DT_MAX)
    if mouse.locked:
        cam_yaw   += mouse.velocity[0] * CAM_SENSITIVITY
        cam_pitch -= mouse.velocity[1] * CAM_SENSITIVITY
        cam_pitch  = clamp(cam_pitch, -89, 89)
    camera.rotation = (cam_pitch, cam_yaw, 0)

    move = Vec3(0, 0, 0)
    if held_keys['w']:     move += camera.forward
    if held_keys['s']:     move -= camera.forward
    if held_keys['d']:     move += camera.right
    if held_keys['a']:     move -= camera.right
    if held_keys['space']: move += Vec3(0, 1, 0)
    if held_keys['q']:     move -= Vec3(0, 1, 0)
    if move.length() > 0:
        speed = CAM_SPEED * (CAM_BOOST if held_keys['shift'] else 1.0)
        camera.position += move.normalized() * speed * dt

def input(key):
    global cam_yaw, cam_pitch
    if key == 'escape':
        mouse.locked = not mouse.locked
    elif key == 'r':
        camera.position = CAM_DEFAULT_POS
        cam_yaw, cam_pitch = CAM_DEFAULT_YAW, CAM_DEFAULT_PITCH

# ====================================================
# ========== 12. 趋化感知（向量化）==================
# ====================================================
food_xy = np.zeros((0, 2))
def refresh_food_array():
    global food_xy
    if foods:
        food_xy = np.array([[f.x, f.z] for f in foods], dtype=float)
    else:
        food_xy = np.zeros((0, 2))

def food_concentration(x, z):
    if food_xy.shape[0] == 0:
        return 0.0
    d2 = (food_xy[:, 0] - x) ** 2 + (food_xy[:, 1] - z) ** 2
    return float(np.exp(-d2 / (2.0 * FOOD_SIGMA ** 2)).sum())

def chemotaxis_probe(pos, angle, probe=PROBE):
    a_l = angle + 0.5
    a_r = angle - 0.5
    xl = pos.x + math.cos(a_l) * probe
    zl = pos.z + math.sin(a_l) * probe
    xr = pos.x + math.cos(a_r) * probe
    zr = pos.z + math.sin(a_r) * probe
    return food_concentration(xl, zl), food_concentration(xr, zr)

def normalize_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

# ====================================================
# ========== 13. 主循环 ==============================
# ====================================================
STEPS_PER_FRAME = 3
LOOK_AT_HEAD_SEGMENTS = 2

def update():
    global V, last_spikes, total_steps, head_pos, head_angle, score
    global first_frame, wave_phase, turn_smoothed

    dt = min(time.dt, DT_MAX)

    update_camera()

    # ---------- 趋化感知 ----------
    refresh_food_array()
    c_left, c_right = chemotaxis_probe(head_pos, head_angle)
    c_scale = 6.0
    drive_left  = np.clip(c_left  * c_scale, 0.0, 1.0)
    drive_right = np.clip(c_right * c_scale, 0.0, 1.0)

    base_sensory = 4.0
    for n in sensory_left:
        I_ext[n] = base_sensory + 6.0 * drive_left
    for n in sensory_right:
        I_ext[n] = base_sensory + 6.0 * drive_right

    grad = c_left - c_right

    # ---------- 神经模拟 ----------
    recent_spikes = np.zeros(N_NEURONS)
    for _ in range(STEPS_PER_FRAME):
        I_syn = np.clip(W @ last_spikes, -50.0, 50.0)
        I_total = I_ext + I_syn
        dV = (-(V - V_rest) + R_m * I_total) / tau_m * dt_sim
        V = V + dV
        V = np.clip(V, V_min, V_max)
        fired = V >= V_thresh
        recent_spikes += fired
        last_spikes = fired.astype(float)
        V[fired] = V_reset
        total_steps += 1

    # ---------- 解码运动指令 ----------
    fwd_count = sum(1 for i in forward_neurons if recent_spikes[i] > 0)
    lft_count = sum(1 for i in left_neurons    if recent_spikes[i] > 0)
    rgt_count = sum(1 for i in right_neurons   if recent_spikes[i] > 0)

    raw_turn = (lft_count - rgt_count) * 0.15
    if grad < 0.02:
        raw_turn += (np.random.rand() - 0.5) * 0.30
    else:
        raw_turn += (np.random.rand() - 0.5) * 0.05

    alpha = 1.0 - math.exp(-dt / TURN_TAU)
    turn_smoothed += (raw_turn - turn_smoothed) * alpha

    head_swing = HEAD_SWING_AMP * math.sin(wave_phase * 0.5)

    head_angle += (turn_smoothed * 0.12 + head_swing * 0.04)
    head_angle = normalize_angle(head_angle)

    speed = 0.6 + BASE_SPEED * (fwd_count / max(1, len(forward_neurons)))
    speed *= (1.0 - 0.3 * max(0.0, grad))

    forward_x = math.cos(head_angle)
    forward_z = math.sin(head_angle)
    head_pos.x += forward_x * speed * dt
    head_pos.z += forward_z * speed * dt

    # ---------- 边界 ----------
    r = math.sqrt(head_pos.x ** 2 + head_pos.z ** 2)
    if r > WORLD_RADIUS - 0.8:
        push = (r - (WORLD_RADIUS - 0.8)) * 0.6
        head_pos.x -= (head_pos.x / r) * push
        head_pos.z -= (head_pos.z / r) * push
        to_center = math.atan2(-head_pos.z, -head_pos.x)
        diff = normalize_angle(to_center - head_angle)
        head_angle = normalize_angle(head_angle + diff * 0.12)

    # ---------- 行波相位 ----------
    wave_freq = WAVE_BASE_SPEED + WAVE_SPEED_GAIN * speed
    wave_phase += wave_freq * dt

    # ---------- 身体链条 ----------
    seg_positions[0] = head_pos
    seg_velocities[0] = Vec3(0, 0, 0)

    if first_frame:
        for i in range(1, N_SEGMENTS):
            seg_positions[i] = Vec3(
                head_pos.x - i * SPACING * forward_x,
                0.1,
                head_pos.z - i * SPACING * forward_z,
            )
        first_frame = False
    else:
        perp_x = -forward_z
        perp_z =  forward_x

        for i in range(1, N_SEGMENTS):
            prev = seg_positions[i - 1]
            curr = seg_positions[i]

            if i == 1:
                body_dir = Vec3(-forward_x, 0, -forward_z)
            else:
                d_prev = seg_positions[i - 1] - seg_positions[i - 2]
                if d_prev.length() > 1e-5:
                    body_dir = d_prev.normalized()
                else:
                    body_dir = Vec3(-forward_x, 0, -forward_z)

            target_dist = prev + body_dir * SPACING

            wave_amp_i = WAVE_AMP_BODY * (WAVE_DECAY_BODY ** (i - 1))
            wave_off = wave_amp_i * math.sin(wave_phase - i * WAVE_FREQ_BODY)

            target_dir = Vec3(
                body_dir.x + perp_x * wave_off,
                0,
                body_dir.z + perp_z * wave_off,
            )
            if target_dir.length() > 1e-5:
                target_dir = target_dir.normalized()
            target_angle = prev + target_dir * SPACING

            target = target_dist * 0.85 + target_angle * 0.15

            to_target = target - curr
            force = to_target * SPRING_K - seg_velocities[i] * DAMPING_C
            seg_velocities[i] += force * dt
            seg_positions[i] = curr + seg_velocities[i] * dt

    for i in range(1, N_SEGMENTS):
        p = seg_positions[i]
        rr = math.sqrt(p.x ** 2 + p.z ** 2)
        if rr > WORLD_RADIUS - 0.5:
            p.x *= (WORLD_RADIUS - 0.5) / rr
            p.z *= (WORLD_RADIUS - 0.5) / rr

    # ---------- 更新 3D ----------
    for i in range(N_SEGMENTS):
        seg = worm_segments[i]
        pos = seg_positions[i]
        y = 0.1 + 0.015 * math.sin(wave_phase - i * WAVE_FREQ_BODY)
        seg.position = Vec3(pos.x, y, pos.z)

        if i < N_SEGMENTS - 1 and i < LOOK_AT_HEAD_SEGMENTS:
            d = seg_positions[i + 1] - seg_positions[i]
            if d.length() > 0.01:
                seg.look_at(seg_positions[i + 1])
        elif i >= LOOK_AT_HEAD_SEGMENTS:
            seg.rotation = (0, 0, 0)

        seg.color = color.rgb(0.95, 0.86, 0.72)

    head = worm_segments[0]
    eye_offset_fwd  = Vec3(forward_x * 0.10, 0.06, forward_z * 0.10)
    eye_offset_perp = Vec3(-forward_z * 0.06, 0, forward_x * 0.06)
    eye_l.position = head.position + eye_offset_fwd + eye_offset_perp
    eye_r.position = head.position + eye_offset_fwd - eye_offset_perp

    # ---------- 食物碰撞 ----------
    for f in foods[:]:
        if (head.position - f.position).length() < 0.35:
            score += 1
            destroy(f)
            foods.remove(f)
            spawn_food()

    # ---------- UI ----------
    active_recent = int(recent_spikes.sum())
    active_last   = int(last_spikes.sum())
    info_text.text = (
        f'C. elegans Virtual Environment\n'
        f'Score: {score}\n'
        f'Active neurons (3 steps): {active_recent}\n'
        f'Instant spikes (last step): {active_last}\n'
        f'Forward / Left / Right: {fwd_count}/{lft_count}/{rgt_count}\n'
        f'Left / Right conc: {c_left:.3f} / {c_right:.3f}  (grad {grad:+.3f})\n'
        f'Mean membrane potential: {V.mean():.2f} mV\n'
        f'Camera: ({camera.x:.1f}, {camera.y:.1f}, {camera.z:.1f})'
    )

app.run()