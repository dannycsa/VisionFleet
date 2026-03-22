#!/usr/bin/env python3
"""
gps_map_node.py — VisionFleet
Pure UI node. Camera feed shows only the selected bike.
Updated with Resizable Window scaling and exact anomaly markers.
"""
import os
import json
import math
import subprocess
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Float32MultiArray
from nav_msgs.msg import Path
from cv_bridge import CvBridge

MAP_IMG_PATH = os.path.expanduser('~/VisionFleet_ws/assets/town10_map.png')
BOUNDS_PATH  = os.path.expanduser('~/VisionFleet_ws/assets/town10_bounds.json')
ZONES_JSON   = os.path.expanduser('~/VisionFleet_ws/assets/active_zones.json')
ASSETS_DIR   = os.path.expanduser('~/VisionFleet_ws/assets')
SPAWNER_PATH = os.path.join(ASSETS_DIR, 'spawn_obstacles.py')
CONTROL_PATH = os.path.join(ASSETS_DIR, 'attach_control.py')

# ── Layout ────────────────────────────────────────────────────
SIDEBAR_W = 360  
BORDER    = 18
WIN_H     = 960  
MAP_W     = 960  
WIN_W     = MAP_W + SIDEBAR_W + BORDER * 2
MAP_X     = BORDER
MAP_Y     = BORDER
MAP_H     = WIN_H - BORDER * 2
SB_X      = WIN_W - SIDEBAR_W
CAM_W     = SIDEBAR_W - 24
CAM_H     = int(CAM_W * 9 / 16)

# Radius matching the map_node spatial clustering
ZONE_WORLD_R = 15.0

PALETTE = [
    (0,   255, 136),
    (0,   180, 216),
    (244, 162,  97),
    (255,  80, 160),
    (160, 100, 255),
    (80,  220, 255),
    (255, 220,  50),
]
_pal_idx  = 0
_pal_lock = threading.Lock()

def _next_color():
    global _pal_idx
    with _pal_lock:
        c = PALETTE[_pal_idx % len(PALETTE)]
        _pal_idx += 1
    return c

COLOR_BG        = (28,  28,  36)
COLOR_BORDER    = (55,  55,  70)
COLOR_SIDEBAR   = (22,  22,  30)
COLOR_DIVIDER   = (50,  50,  65)
COLOR_TEXT      = (200, 200, 210)
COLOR_TEXT_DIM  = (110, 110, 128)
COLOR_BARRIER   = (255, 102,   0)
COLOR_BTN       = (45,  45,  60)
COLOR_BTN_HOV   = (60,  60,  80)
COLOR_SEL       = (55,  55,  75)
ARROW_LENGTH    = 18
ARROW_WIDTH     = 10

STATUS_COLORS = {
    'navigating':         (0,   220, 100),
    'stopped_at_barrier': (255,  80,  80),
    'rerouting':          (255, 200,   0),
    'waiting_at_goal':    (100, 180, 255),
    'waiting':            (130, 130, 140),
}


def world_to_map(x, y, bounds):
    fx = (x - bounds['min_x']) / (bounds['max_x'] - bounds['min_x'])
    fy = (y - bounds['min_y']) / (bounds['max_y'] - bounds['min_y'])
    sx = int(MAP_X + fx * MAP_W)
    sy = int(MAP_Y + fy * MAP_H)
    return (max(MAP_X, min(MAP_X + MAP_W - 1, sx)),
            max(MAP_Y, min(MAP_Y + MAP_H - 1, sy)))


def metres_to_px(metres, bounds):
    span = bounds['max_x'] - bounds['min_x']
    return max(4, int(metres / span * MAP_W))


def make_arrowhead(cx, cy, yaw_deg, length, half_width):
    a        = math.radians(yaw_deg)
    fx, fy   =  math.cos(a),  math.sin(a)
    sx, sy_a = -math.sin(a),  math.cos(a)
    return [
        (int(cx + fx * length),     int(cy + fy * length)),
        (int(cx + sx * half_width), int(cy + sy_a * half_width)),
        (int(cx - sx * half_width), int(cy - sy_a * half_width)),
    ]


def zone_colors(confidence, bike_count=1):
    if bike_count < 2 or confidence < 0.45:
        return (60, 220, 100), 55, 180   # green
    elif confidence < 0.70:
        return (255, 210, 60), 65, 190   # yellow
    else:
        return (255, 80, 80), 90, 210    # red

def load_active_zones():
    try:
        if os.path.exists(ZONES_JSON):
            with open(ZONES_JSON) as f:
                return json.load(f)
    except Exception:
        pass
    return []


class BikeDisplay:
    def __init__(self, bike_id, color):
        self.bike_id = bike_id
        self.color   = color
        self.x = self.y = None
        self.yaw    = 0.0
        self.status = 'waiting'
        self.goal   = None
        self.path   = []


class Button:
    def __init__(self, x, y, w, h, label, color=None):
        self.rect  = (x, y, w, h)
        self.label = label
        self.color = color or COLOR_BTN

    def draw(self, surface, font, mouse_pos, override_color=None):
        import pygame
        x, y, w, h = self.rect
        col = override_color or (
            COLOR_BTN_HOV if self._hovered(mouse_pos) else self.color)
        pygame.draw.rect(surface, col,          self.rect, border_radius=4)
        pygame.draw.rect(surface, COLOR_DIVIDER, self.rect, 1, border_radius=4)
        txt = font.render(self.label, True, COLOR_TEXT)
        surface.blit(txt, (x + (w - txt.get_width())  // 2,
                           y + (h - txt.get_height()) // 2))

    def _hovered(self, mp):
        x, y, w, h = self.rect
        return x <= mp[0] <= x+w and y <= mp[1] <= y+h

    def clicked(self, mp, event):
        import pygame
        return (event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self._hovered(mp))


class GpsMapNode(Node):

    def __init__(self):
        super().__init__('gps_map_node')

        self._lock          = threading.Lock()
        self._bikes         = {}
        self._obstacles     = []
        self._zones         = []
        self._selected_bike = None

        self._bridge      = CvBridge()
        self._cam_frames  = {}
        self._cam_lock    = threading.Lock()
        self._cam_subs    = set()

        self._ctrl_proc   = None
        self._obs_proc    = None
        self._obs_count   = 5

        self.create_subscription(
            Float32MultiArray, '/visionfleet/world/obstacles',
            self._cb_obstacles, 10)

        self._bounds = self._load_bounds()
        self.create_timer(2.0, self._discover_bikes)
        self.create_timer(1.0, self._refresh_zones) 

        threading.Thread(target=self._run_pygame, daemon=True).start()
        self.get_logger().info('GPS map UI node ready.')

    def _discover_bikes(self):
        known = [n for n, _ in self.get_topic_names_and_types()]
        for topic in known:
            if topic.startswith('/visionfleet/') and topic.endswith('/fused_pose'):
                parts = topic.split('/')
                if len(parts) == 4:
                    with self._lock:
                        self._register_bike(parts[2])

    def _register_bike(self, bid):
        if bid in self._bikes:
            return
        color = _next_color()
        self._bikes[bid] = BikeDisplay(bid, color)
        if self._selected_bike is None:
            self._selected_bike = bid

        self.create_subscription(
            PointStamped, f'/visionfleet/{bid}/fused_pose',
            self._make_pose_cb(bid), 10)
        self.create_subscription(
            Path,         f'/visionfleet/{bid}/planned_path',
            self._make_path_cb(bid), 10)
        self.create_subscription(
            String,       f'/visionfleet/{bid}/nav_status',
            self._make_status_cb(bid), 10)
        self.create_subscription(
            PointStamped, f'/visionfleet/{bid}/goal',
            self._make_goal_cb(bid), 10)

        self._subscribe_camera(bid)
        self.get_logger().info(f'[map UI] Registered bike: {bid}')

    def _subscribe_camera(self, bid):
        if bid in self._cam_subs:
            return
        self._cam_subs.add(bid)
        self.create_subscription(
            Image,
            f'/visionfleet/{bid}/detection_image',
            self._make_cam_cb(bid),
            1 
        )

    def _make_cam_cb(self, bid):
        def cb(msg: Image):
            try:
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                with self._cam_lock:
                    self._cam_frames[bid] = frame
            except Exception:
                pass
        return cb

    def _make_pose_cb(self, bid):
        def cb(msg):
            with self._lock:
                b = self._bikes[bid]
                b.x   = msg.point.x
                b.y   = msg.point.y
                b.yaw = msg.point.z
        return cb

    def _make_path_cb(self, bid):
        def cb(msg):
            pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
            with self._lock:
                self._bikes[bid].path = pts
        return cb

    def _make_status_cb(self, bid):
        def cb(msg):
            with self._lock:
                self._bikes[bid].status = msg.data
        return cb

    def _make_goal_cb(self, bid):
        def cb(msg):
            with self._lock:
                self._bikes[bid].goal = (msg.point.x, msg.point.y)
        return cb

    def _cb_obstacles(self, msg):
        data = msg.data
        obs  = []
        for i in range(0, len(data), 4):
            obs.append((data[i], data[i+1], data[i+2], data[i+3]))
        with self._lock:
            self._obstacles = obs

    def _refresh_zones(self):
        with self._lock:
            self._zones = load_active_zones()

    def _load_bounds(self):
        if os.path.exists(BOUNDS_PATH):
            with open(BOUNDS_PATH) as f:
                b = json.load(f)
            return b
        return {'min_x': -124.6, 'max_x': 120.0,
                'min_y':  -78.7, 'max_y': 151.2}

    def _ctrl_running(self):
        return self._ctrl_proc is not None and self._ctrl_proc.poll() is None

    def _obs_running(self):
        return self._obs_proc is not None and self._obs_proc.poll() is None

    def _launch_control(self, bike_id):
        if self._ctrl_running():
            self._ctrl_proc.terminate()
            self._ctrl_proc = None
            return
        try:
            self._ctrl_proc = subprocess.Popen(
                ['python3', CONTROL_PATH, '--bike', bike_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            self.get_logger().info(f'Manual control started: {bike_id}')
        except Exception as e:
            self.get_logger().warn(f'attach_control failed: {e}')
            self._ctrl_proc = None

    def _spawn_obstacles(self):
        if self._obs_running():
            return
        try:
            self._obs_proc = subprocess.Popen(
                ['python3', SPAWNER_PATH, '-n', str(self._obs_count)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            self.get_logger().info(f'Spawning {self._obs_count} obstacle blocks')
        except Exception as e:
            self.get_logger().warn(f'spawn_obstacles failed: {e}')

    def _remove_obstacles(self):
        if self._obs_running():
            self._obs_proc.terminate()
            self._obs_proc = None
        try:
            subprocess.Popen(
                ['python3', SPAWNER_PATH, '--remove'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            self.get_logger().info('Removing all obstacles')
        except Exception as e:
            self.get_logger().warn(f'remove obstacles failed: {e}')

    def _cleanup_subprocesses(self):
        if self._ctrl_running():
            self._ctrl_proc.terminate()
        if self._obs_running():
            self._obs_proc.terminate()

    def _run_pygame(self):
        import pygame
        import cv2
        pygame.init()
        
        # Configure window to be RESIZABLE
        screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
        pygame.display.set_caption('VisionFleet — GPS Map')
        clock = pygame.time.Clock()
        
        # We draw everything to this virtual canvas, then scale it to the physical screen
        canvas = pygame.Surface((WIN_W, WIN_H))
        
        current_w, current_h = WIN_W, WIN_H
        
        f_title = pygame.font.SysFont('monospace', 16, bold=True)
        f_label = pygame.font.SysFont('monospace', 14)
        f_small = pygame.font.SysFont('monospace', 12)

        if os.path.exists(MAP_IMG_PATH):
            img      = cv2.cvtColor(cv2.imread(MAP_IMG_PATH), cv2.COLOR_BGR2RGB)
            img      = cv2.resize(img, (MAP_W, MAP_H), interpolation=cv2.INTER_AREA)
            map_surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        else:
            map_surf = pygame.Surface((MAP_W, MAP_H))
            map_surf.fill((80, 80, 80))

        zone_surf = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
        path_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)

        btn_ctrl   = Button(0, 0, SIDEBAR_W - 24, 30, 'manual control')
        btn_spawn  = Button(0, 0, SIDEBAR_W - 100, 30, 'spawn')
        btn_remove = Button(0, 0, 70, 30, 'remove', color=(80, 30, 30))
        btn_obs_m  = Button(0, 0, 26, 30, '-')
        btn_obs_p  = Button(0, 0, 26, 30, '+')

        bike_row_rects = {}

        while rclpy.ok():
            # Handle Mouse scaling for resizable window
            phys_mouse = pygame.mouse.get_pos()
            mouse = (
                int(phys_mouse[0] * (WIN_W / current_w)),
                int(phys_mouse[1] * (WIN_H / current_h))
            )
            
            events = pygame.event.get()

            for ev in events:
                if ev.type == pygame.QUIT:
                    self._cleanup_subprocesses()
                    return
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    self._cleanup_subprocesses()
                    return
                if ev.type == pygame.VIDEORESIZE:
                    current_w, current_h = ev.w, ev.h
                    screen = pygame.display.set_mode((current_w, current_h), pygame.RESIZABLE)

                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    for bid, rect in bike_row_rects.items():
                        rx, ry, rw, rh = rect
                        if rx <= mouse[0] <= rx+rw and ry <= mouse[1] <= ry+rh:
                            with self._lock:
                                self._selected_bike = bid

                if btn_ctrl.clicked(mouse, ev):
                    with self._lock:
                        sel = self._selected_bike
                    if sel:
                        self._launch_control(sel)

                if btn_obs_m.clicked(mouse, ev):
                    self._obs_count = max(1, self._obs_count - 1)
                if btn_obs_p.clicked(mouse, ev):
                    self._obs_count = min(10, self._obs_count + 1)
                if btn_spawn.clicked(mouse, ev):
                    self._spawn_obstacles()
                if btn_remove.clicked(mouse, ev):
                    self._remove_obstacles()

            with self._lock:
                bikes_snap = {
                    bid: (b.x, b.y, b.yaw, b.color, b.status, b.goal, list(b.path))
                    for bid, b in self._bikes.items()
                }
                obstacles = list(self._obstacles)
                
                zones = list(self._zones)
                zones.sort(key=lambda z: z.get('confidence', 0), reverse=True)
                
                selected  = self._selected_bike

            with self._cam_lock:
                cam_frame = self._cam_frames.get(selected, None)

            bounds = self._bounds

            # ── Map (Draw to Virtual Canvas) ──────────────────
            canvas.fill(COLOR_BG)
            pygame.draw.rect(canvas, COLOR_BORDER,
                (MAP_X-2, MAP_Y-2, MAP_W+4, MAP_H+4), 2, border_radius=3)
            canvas.blit(map_surf, (MAP_X, MAP_Y))

            zone_surf.fill((0, 0, 0, 0))
            zone_r = metres_to_px(ZONE_WORLD_R, bounds)
            for zone in zones:
                zid  = zone.get('id', '?')
                lat  = zone.get('lat', 0)
                lon  = zone.get('lon', 0)
                conf = zone.get('confidence', 0.4)
                
                wx   = lon * (6378137.0 * math.pi / 180.0)
                wy   = lat * (6378137.0 * math.pi / 180.0)
                sx, sy = world_to_map(wx, wy, bounds)
                zx, zy = sx - MAP_X, sy - MAP_Y
                
                n_b = zone.get('bike_count', 1)
                col, fill_a, border_a = zone_colors(conf, n_b)
                
                pygame.draw.circle(zone_surf, (*col, fill_a),   (zx, zy), zone_r)
                pygame.draw.circle(zone_surf, (*col, border_a), (zx, zy), zone_r, 2)
                
                # Draw #ID exactly on the centroid (No X)
                id_txt = f_small.render(f'#{zid}', True, (255, 255, 255, 220))
                txt_w = id_txt.get_width()
                txt_h = id_txt.get_height()
                zone_surf.blit(id_txt, (zx - txt_w // 2, zy - txt_h // 2))
                
            canvas.blit(zone_surf, (MAP_X, MAP_Y))

            path_surf.fill((0, 0, 0, 0))
            for bid, (x, y, yaw, color, status, goal, path) in bikes_snap.items():
                if len(path) < 2:
                    continue
                r, g, bv   = color
                pts_sub    = path[::3]
                screen_pts = [world_to_map(px, py, bounds) for px, py in pts_sub]
                if len(screen_pts) >= 2:
                    pygame.draw.lines(path_surf, (r, g, bv, 70), False, screen_pts, 3)
            canvas.blit(path_surf, (0, 0))

            for ox, oy, yaw, lw in obstacles:
                rad = math.radians(yaw)
                hl  = lw * 0.45
                sx1, sy1 = world_to_map(
                    ox - math.cos(rad)*hl, oy - math.sin(rad)*hl, bounds)
                sx2, sy2 = world_to_map(
                    ox + math.cos(rad)*hl, oy + math.sin(rad)*hl, bounds)
                pygame.draw.line(canvas, (255,255,255), (sx1,sy1), (sx2,sy2), 8)
                pygame.draw.line(canvas, COLOR_BARRIER,  (sx1,sy1), (sx2,sy2), 4)

            for bid, (x, y, yaw, color, status, goal, path) in bikes_snap.items():
                if goal is None:
                    continue
                gsx, gsy = world_to_map(goal[0], goal[1], bounds)
                r, g, bv = color
                pygame.draw.circle(canvas, (r,g,bv), (gsx,gsy), 12, 2)
                pygame.draw.circle(canvas, (r,g,bv), (gsx,gsy), 4)
                arm = 16
                pygame.draw.line(canvas,(r,g,bv),(gsx-arm,gsy),(gsx-10,gsy),2)
                pygame.draw.line(canvas,(r,g,bv),(gsx+10,gsy),(gsx+arm,gsy),2)
                pygame.draw.line(canvas,(r,g,bv),(gsx,gsy-arm),(gsx,gsy-10),2)
                pygame.draw.line(canvas,(r,g,bv),(gsx,gsy+10),(gsx,gsy+arm),2)

            for bid, (x, y, yaw, color, status, goal, path) in bikes_snap.items():
                if x is None:
                    continue
                sx, sy = world_to_map(x, y, bounds)
                verts  = make_arrowhead(sx, sy, yaw, ARROW_LENGTH, ARROW_WIDTH)
                pygame.draw.polygon(canvas, color, verts)
                pygame.draw.polygon(canvas, (255,255,255), verts, 1)

            # ── Sidebar (Draw to Virtual Canvas) ──────────────
            pygame.draw.rect(canvas, COLOR_SIDEBAR,
                pygame.Rect(SB_X, 0, SIDEBAR_W, WIN_H))
            pygame.draw.line(canvas, COLOR_DIVIDER, (SB_X,0), (SB_X,WIN_H), 1)

            px  = SB_X + 16
            icx = SB_X + 16
            py  = 16

            canvas.blit(f_title.render('VisionFleet', True, (220,220,235)), (px, py))
            py += 22
            canvas.blit(f_small.render('GPS  Monitor', True, COLOR_TEXT_DIM), (px, py))
            py += 24
            pygame.draw.line(canvas, COLOR_DIVIDER, (SB_X+8,py), (WIN_W-8,py), 1)
            py += 12

            # Bikes
            canvas.blit(f_small.render('BIKES', True, COLOR_TEXT_DIM), (px, py))
            py += 18
            bike_row_rects.clear()

            for bid, (x, y, yaw, color, status, goal, path) in bikes_snap.items():
                is_sel = (bid == selected)
                row_h  = 48
                if is_sel:
                    pygame.draw.rect(canvas, COLOR_SEL,
                        (SB_X+6, py-2, SIDEBAR_W-12, row_h), border_radius=4)
                bike_row_rects[bid] = (SB_X+6, py-2, SIDEBAR_W-12, row_h)

                r, g, bv = color
                pygame.draw.polygon(canvas, color,
                    [(icx+9, py+4), (icx+1, py+17), (icx+17, py+17)])
                pygame.draw.polygon(canvas, (255,255,255),
                    [(icx+9, py+4), (icx+1, py+17), (icx+17, py+17)], 1)

                canvas.blit(f_label.render(bid, True, color), (icx+24, py+2))
                stat_color = STATUS_COLORS.get(status, COLOR_TEXT_DIM)
                canvas.blit(f_small.render(
                    status.replace('_',' '), True, stat_color), (icx+24, py+18))
                coord = f'{x:+.0f} {y:+.0f}' if x is not None else 'waiting...'
                canvas.blit(f_small.render(coord, True, COLOR_TEXT_DIM),
                            (icx+24, py+32))
                py += row_h + 6

            py += 4
            pygame.draw.line(canvas, COLOR_DIVIDER, (SB_X+8,py), (WIN_W-8,py), 1)
            py += 12

            # Map elements
            canvas.blit(f_small.render('MAP ELEMENTS', True, COLOR_TEXT_DIM), (px, py))
            py += 18
            pygame.draw.line(canvas,(255,255,255),(icx,py+8),(icx+28,py+8),7)
            pygame.draw.line(canvas,COLOR_BARRIER,(icx,py+8),(icx+28,py+8),4)
            canvas.blit(f_label.render('closed street', True, COLOR_TEXT),(icx+36,py))
            canvas.blit(f_small.render(f'{len(obstacles)} active',
                True, COLOR_TEXT_DIM), (icx+36, py+18))
            py += 36

            pygame.draw.line(canvas, COLOR_DIVIDER, (SB_X+8,py), (WIN_W-8,py), 1)
            py += 12

            # Event zones (Sorted)
            canvas.blit(f_small.render('EVENT ZONES', True, COLOR_TEXT_DIM), (px, py))
            py += 18
            if zones:
                for zone in zones[:4]:
                    zid   = zone.get('id', '?')
                    conf  = zone.get('confidence', 0.4)
                    n_b   = zone.get('bike_count', 1)
                    col, _, _ = zone_colors(conf, n_b)
                    
                    ghost = pygame.Surface((28,28), pygame.SRCALPHA)
                    pygame.draw.circle(ghost, (*col,130), (14,14), 10)
                    pygame.draw.circle(ghost, (*col,220), (14,14), 10, 2)
                    canvas.blit(ghost, (icx, py))
                    
                    canvas.blit(f_label.render(f'#{zid}   {int(conf*100)}%', True, col),
                                (icx+36, py+2))
                    canvas.blit(f_small.render(f'{n_b} bike(s) validated', True, COLOR_TEXT_DIM),
                                (icx+36, py+18))
                    py += 36
                    
                if len(zones) > 4:
                    canvas.blit(f_small.render(f'+{len(zones)-4} more',
                        True, COLOR_TEXT_DIM), (px, py))
                    py += 18
            else:
                canvas.blit(f_small.render('none active', True, COLOR_TEXT_DIM),(px,py))
                py += 18

            pygame.draw.line(canvas, COLOR_DIVIDER, (SB_X+8,py), (WIN_W-8,py), 1)
            py += 12

            # Camera feed
            canvas.blit(f_small.render(
                f'CAMERA  [{selected or "none"}]', True, COLOR_TEXT_DIM), (px, py))
            py += 18
            cam_rect = (SB_X + 12, py, CAM_W, CAM_H)
            pygame.draw.rect(canvas, (18,18,26), cam_rect)
            pygame.draw.rect(canvas, COLOR_DIVIDER, cam_rect, 1)

            if cam_frame is not None:
                try:
                    import cv2
                    resized   = cv2.resize(cam_frame, (CAM_W, CAM_H),
                                           interpolation=cv2.INTER_LINEAR)
                    cam_surf  = pygame.surfarray.make_surface(
                        np.transpose(resized, (1,0,2)))
                    canvas.blit(cam_surf, (SB_X+12, py))
                except Exception:
                    pass
            else:
                ns = f_small.render('no signal', True, COLOR_TEXT_DIM)
                canvas.blit(ns, (SB_X+12+(CAM_W-ns.get_width())//2,
                                 py+(CAM_H-ns.get_height())//2))
            py += CAM_H + 12

            pygame.draw.line(canvas, COLOR_DIVIDER, (SB_X+8,py), (WIN_W-8,py), 1)
            py += 12

            # Controls
            canvas.blit(f_small.render('CONTROLS', True, COLOR_TEXT_DIM), (px, py))
            py += 18
            btn_ctrl.rect  = (SB_X+12, py, SIDEBAR_W-24, 30)
            btn_ctrl.label = 'stop control' if self._ctrl_running() else 'manual control'
            ctrl_col = (30,110,55) if self._ctrl_running() else COLOR_BTN
            btn_ctrl.draw(canvas, f_small, mouse, override_color=ctrl_col)
            py += 40

            # Obstacles
            canvas.blit(f_small.render('OBSTACLES', True, COLOR_TEXT_DIM), (px, py))
            py += 18
            btn_obs_m.rect = (SB_X+12,        py, 26, 30)
            btn_obs_m.draw(canvas, f_label, mouse)
            count_lbl = f_label.render(str(self._obs_count), True, COLOR_TEXT)
            canvas.blit(count_lbl, (SB_X+48, py+6))
            btn_obs_p.rect = (SB_X+72, py, 26, 30)
            btn_obs_p.draw(canvas, f_label, mouse)

            dot_col = (60,200,80) if self._obs_running() else COLOR_TEXT_DIM
            dot_txt = f_small.render(
                '● running' if self._obs_running() else '● idle',
                True, dot_col)
            canvas.blit(dot_txt, (SB_X+108, py+8))
            py += 40

            spawn_w        = SIDEBAR_W - 24 - 76
            btn_spawn.rect  = (SB_X+12,               py, spawn_w, 30)
            btn_remove.rect = (SB_X+12+spawn_w+6,     py, 70,      30)
            spawn_col = (40,40,50) if self._obs_running() else COLOR_BTN
            btn_spawn.draw(canvas,  f_small, mouse, override_color=spawn_col)
            btn_remove.draw(canvas, f_small, mouse)
            py += 42

            canvas.blit(f_small.render(f'fps {clock.get_fps():.0f}',
                True, COLOR_TEXT_DIM), (SB_X+12, WIN_H-24))

            # ── Scale Canvas to Physical Screen ───────────────
            scaled_canvas = pygame.transform.smoothscale(canvas, (current_w, current_h))
            screen.blit(scaled_canvas, (0, 0))
            
            pygame.display.flip()
            clock.tick(20)

        pygame.quit()

    def destroy_node(self):
        self._cleanup_subprocesses()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GpsMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()