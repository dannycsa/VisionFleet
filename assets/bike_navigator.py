#!/usr/bin/env python3
"""
bike_navigator_node.py — VisionFleet
=====================================
Autonomous navigation for all spawned bikes.
Run AFTER fleet_spawner.py has spawned the vehicles.

- Discovers bikes by scanning CARLA actors (same pattern as fusion_node)
- One tick loop, one CARLA connection, one ROS node
- Per-bike state machines: navigating → stopped → rerouting → waiting
- Goals are nearest road waypoints to building/block locations
- Visual goal published as building location for map display
- Path planner ignores blocked road data — bikes discover organically
- Detection events published when stopped at barrier → feeds map_node

Usage:
  python3 bike_navigator_node.py
  python3 bike_navigator_node.py --host localhost --port 2000
"""

import json
import logging
import math
import os
import random
import threading
import time

import carla
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix

# ── Navigation tuning ─────────────────────────────────────────
WAYPOINT_STEP        = 2.0    # metres between path waypoints
GOAL_REACHED_DIST    = 10.0   # metres — "close enough" to goal
BARRIER_DETECT_DIST  = 14.0   # metres — forward cone check distance
STOP_DURATION        = 4.0    # seconds stopped at barrier before replanning
WAIT_AT_GOAL         = 3.0    # seconds waiting at delivery goal
PATH_LOOKAHEAD       = 60     # waypoints fed to traffic manager at once
PATH_MAX_LEN         = 600    # max waypoints in a path
SPEED_NORMAL         = 20.0   # km/h
UTURN_REVERSE_TIME   = 2.5    # seconds of reverse for U-turn
BUILDING_OFFSET      = 9.0    # metres off road to place visual goal marker
TICK_DT              = 0.05   # seconds per world tick (matches fixed_delta_seconds)
DETECT_INTERVAL      = 1.0    # seconds between detection publishes at barrier
REDISCOVER_INTERVAL  = 5.0    # seconds between bike discovery scans

EARTH_R = 6378137.0


# ── State labels ──────────────────────────────────────────────
S_NAV     = 'navigating'
S_STOPPED = 'stopped_at_barrier'
S_REROUTE = 'rerouting'
S_UTURN   = 'u_turn'
S_WAIT    = 'waiting_at_goal'


# ── Helpers ───────────────────────────────────────────────────

def dist2d(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def barrier_ahead(vehicle_tf, barriers, detect_dist):
    """True if any barrier is within detect_dist AND in front of the vehicle."""
    yaw   = math.radians(vehicle_tf.rotation.yaw)
    fwd_x = math.cos(yaw)
    fwd_y = math.sin(yaw)
    vx    = vehicle_tf.location.x
    vy    = vehicle_tf.location.y
    for b in barriers:
        dx = b.x - vx
        dy = b.y - vy
        d  = math.sqrt(dx**2 + dy**2)
        if d > detect_dist:
            continue
        # Dot product > 0 means barrier is in the forward half-space
        if dx * fwd_x + dy * fwd_y > 0:
            return True
    return False


def get_barrier_locations(world):
    return [a.get_transform().location
            for a in world.get_actors()
            if 'streetbarrier' in a.type_id]


def path_clear_of_barriers(path, barriers, clearance=5.5):
    """Return True if no waypoint in path is within clearance of any barrier."""
    for wp in path:
        loc = wp.transform.location
        for b in barriers:
            if dist2d(loc, b) < clearance:
                return False
    return True


def build_path(cmap, start_loc, goal_wp, max_len=PATH_MAX_LEN):
    """Greedy path from start_loc toward goal_wp following road topology."""
    current = cmap.get_waypoint(
        start_loc, project_to_road=True,
        lane_type=carla.LaneType.Driving)
    if current is None:
        return []
    path    = [current]
    visited = {current.id}
    goal_loc = goal_wp.transform.location
    for _ in range(max_len):
        nexts = current.next(WAYPOINT_STEP)
        if not nexts:
            break
        best = min(nexts, key=lambda w: dist2d(w.transform.location, goal_loc))
        if best.id in visited:
            break
        visited.add(best.id)
        path.append(best)
        current = best
        if dist2d(current.transform.location, goal_loc) < GOAL_REACHED_DIST:
            break
    return path


def reroute_around(cmap, veh_loc, goal_wp, barriers):
    """
    Try to find an alternate path that avoids known barriers.
    Strategies (in order):
      1. Step back N waypoints then try left lane
      2. Step back N waypoints on current lane
      3. Random mid-point detour
    Returns a path list, or [] if nothing found.
    """
    start_wp = cmap.get_waypoint(
        veh_loc, project_to_road=True,
        lane_type=carla.LaneType.Driving)
    if start_wp is None:
        return []

    for step_back in [6, 12, 24]:
        prevs = start_wp.previous(step_back * WAYPOINT_STEP)
        if not prevs:
            continue
        back_wp    = prevs[-1]
        candidates = [back_wp]
        left = back_wp.get_left_lane()
        if left and left.lane_type == carla.LaneType.Driving:
            candidates.append(left)
        for cand in candidates:
            alt = build_path(cmap, cand.transform.location, goal_wp)
            if alt and path_clear_of_barriers(alt, barriers):
                return alt

    # Random detour via mid-point
    for _ in range(12):
        nexts = start_wp.next(random.uniform(25, 80))
        if not nexts:
            continue
        mid = random.choice(nexts)
        if all(dist2d(mid.transform.location, b) > 14.0 for b in barriers):
            alt = build_path(cmap, mid.transform.location, goal_wp)
            if alt and path_clear_of_barriers(alt, barriers):
                return alt
    return []


def find_delivery_goals(cmap, count=50):
    """
    Find waypoints that are adjacent to sidewalks (near buildings).
    Returns list of (road_waypoint, visual_location) tuples:
      - road_waypoint: actual navigation target (on the road)
      - visual_location: carla.Location offset ~BUILDING_OFFSET m from road
                         toward the sidewalk — shown on map as the 'house'
    """
    goals = []
    seen  = set()

    for wp in cmap.generate_waypoints(8.0):
        if wp.lane_type != carla.LaneType.Driving:
            continue
        right = wp.get_right_lane()
        if not (right and right.lane_type == carla.LaneType.Sidewalk):
            continue
        if wp.road_id in seen:
            continue
        seen.add(wp.road_id)

        # Visual goal: project perpendicularly toward the sidewalk
        yaw   = math.radians(wp.transform.rotation.yaw)
        # Right-perpendicular in CARLA's coordinate system
        rx    = math.sin(yaw)
        ry    = -math.cos(yaw)
        loc   = wp.transform.location
        visual_loc = carla.Location(
            x=loc.x + rx * BUILDING_OFFSET,
            y=loc.y + ry * BUILDING_OFFSET,
            z=loc.z)

        goals.append((wp, visual_loc))
        if len(goals) >= count:
            break

    # Fallback: use spawn points if not enough sidewalk goals found
    if len(goals) < 10:
        for sp in cmap.get_spawn_points():
            wp = cmap.get_waypoint(sp.location, project_to_road=True)
            if wp:
                goals.append((wp, sp.location))

    random.shuffle(goals)
    return goals


# ── Per-bike state ────────────────────────────────────────────

class BikeNav:
    """
    All mutable state for one bike's navigation.
    Mutated only from the main tick loop — no locking needed.
    """
    def __init__(self, bike_id, vehicle, color):
        self.bike_id      = bike_id
        self.vehicle      = vehicle
        self.color        = color

        # Navigation state
        self.state        = S_NAV
        self.path         = []          # list of carla.Waypoint
        self.road_goal    = None        # carla.Waypoint — actual nav target
        self.visual_goal  = None        # carla.Location — shown on map

        # Timers (in seconds, accumulated via TICK_DT)
        self.stop_timer   = 0.0
        self.wait_timer   = 0.0
        self.uturn_timer  = 0.0

        # GPS (updated from ROS callback)
        self.lat          = 0.0
        self.lon          = 0.0
        self._gps_lock    = threading.Lock()

        # Last time a detection event was published
        self.last_detect  = 0.0


# ── ROS node ──────────────────────────────────────────────────

class BikeNavigatorNode(Node):
    """
    Single ROS 2 node managing all bikes.
    Publishers/subscribers are created per-bike dynamically.
    """

    def __init__(self):
        super().__init__('bike_navigator_node')
        self._pub_cache = {}   # bike_id -> dict of publishers

    def ensure_publishers(self, bike_id):
        if bike_id in self._pub_cache:
            return
        ns = f'/visionfleet/{bike_id}'
        self._pub_cache[bike_id] = {
            'path':      self.create_publisher(Path,         f'{ns}/planned_path', 10),
            'status':    self.create_publisher(String,       f'{ns}/nav_status',   10),
            'goal':      self.create_publisher(PointStamped, f'{ns}/goal',         10),
            'detection': self.create_publisher(String,       f'{ns}/detections',   10),
        }

    def ensure_gnss_sub(self, bike_id, callback):
        """Subscribe to the correct GNSS topic for this bike."""
        # Our fleet uses /carla/hero/bike_N/gnss (sensor namespaced under hero)
        topic = f'/carla/hero/{bike_id}/gnss'
        self.create_subscription(NavSatFix, topic, callback, 10)
        self.get_logger().info(f'[{bike_id}] GNSS ← {topic}')

    def pub_status(self, bike_id, status):
        msg      = String()
        msg.data = status
        self._pub_cache[bike_id]['status'].publish(msg)

    def pub_goal(self, bike_id, road_wp, visual_loc):
        """Publish the VISUAL goal location (building/block, not road)."""
        msg                 = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.point.x         = visual_loc.x
        msg.point.y         = visual_loc.y
        msg.point.z         = visual_loc.z
        self._pub_cache[bike_id]['goal'].publish(msg)

    def pub_path(self, bike_id, waypoints):
        msg                 = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp    = self.get_clock().now().to_msg()
        for wp in waypoints:
            ps                   = PoseStamped()
            ps.header            = msg.header
            ps.pose.position.x   = wp.transform.location.x
            ps.pose.position.y   = wp.transform.location.y
            ps.pose.position.z   = wp.transform.location.z
            msg.poses.append(ps)
        self._pub_cache[bike_id]['path'].publish(msg)

    def pub_detection(self, bike_id, lat, lon, n_barriers):
        payload = {
            'timestamp':    int(time.time()),
            'road_blocked': True,
            'bike_id':      bike_id,
            'lat':          lat,
            'lon':          lon,
            'count':        n_barriers,
        }
        msg      = String()
        msg.data = json.dumps(payload)
        self._pub_cache[bike_id]['detection'].publish(msg)


# ── Main ──────────────────────────────────────────────────────

def discover_bikes(world):
    """
    Find all vehicles spawned by fleet_spawner.py.
    They have role_name='hero'. We identify which bike_N each one is
    by looking at attached sensors with ros_name like 'bike_1/gnss'.
    Returns list of (bike_id, vehicle_actor).
    """
    all_actors = world.get_actors()
    found      = []

    for actor in all_actors.filter('vehicle.*'):
        if actor.attributes.get('role_name', '') != 'hero':
            continue

        # Find attached sensors
        for sensor in all_actors:
            rn = sensor.attributes.get('ros_name', '')
            if '/' not in rn or 'gnss' not in rn:
                continue
            # Check if this sensor is attached to this vehicle
            # CARLA sensors have a parent_id attribute
            parent_id = getattr(sensor, 'parent', None)
            if parent_id is None:
                continue
            try:
                if sensor.parent.id != actor.id:
                    continue
            except Exception:
                continue
            bike_id = rn.split('/')[0]   # 'bike_1/gnss' → 'bike_1'
            found.append((bike_id, actor))
            break   # one match per vehicle is enough

    return found


def assign_new_goal(bike, tm, goals, ros_node):
    """Pick a random goal, build path, hand to traffic manager."""
    road_wp, visual_loc = random.choice(goals)
    bike.road_goal  = road_wp
    bike.visual_goal = visual_loc

    veh_loc    = bike.vehicle.get_transform().location
    bike.path  = build_path(
        bike.vehicle.get_world().get_map(), veh_loc, road_wp)

    if bike.path:
        locs = [wp.transform.location for wp in bike.path[:PATH_LOOKAHEAD]]
        tm.set_path(bike.vehicle, locs)
        tm.set_desired_speed(bike.vehicle, SPEED_NORMAL)
        bike.vehicle.set_autopilot(True, tm.get_port())
        ros_node.pub_path(bike.bike_id, bike.path)

    ros_node.pub_goal(bike.bike_id, road_wp, visual_loc)
    ros_node.pub_status(bike.bike_id, S_NAV)
    bike.state      = S_NAV
    bike.stop_timer = 0.0
    bike.wait_timer = 0.0

    logging.info(
        f'[{bike.bike_id}] → new goal '
        f'({road_wp.transform.location.x:.0f}, '
        f'{road_wp.transform.location.y:.0f})')


def tick_bike(bike, world, tm, goals, ros_node, barriers):
    """
    Advance one bike's state machine by one tick.
    Called from the main loop for every bike every tick.
    """
    veh_tf  = bike.vehicle.get_transform()
    veh_loc = veh_tf.location

    if bike.state == S_NAV:
        # ── Goal reached? ─────────────────────────────────────
        if dist2d(veh_loc, bike.road_goal.transform.location) < GOAL_REACHED_DIST:
            logging.info(f'[{bike.bike_id}] Goal reached — waiting')
            bike.vehicle.set_autopilot(False, tm.get_port())
            bike.vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, brake=1.0))
            bike.wait_timer = 0.0
            bike.state      = S_WAIT
            ros_node.pub_status(bike.bike_id, S_WAIT)
            return

        # ── Barrier ahead? ────────────────────────────────────
        if barriers and barrier_ahead(veh_tf, barriers, BARRIER_DETECT_DIST):
            logging.info(f'[{bike.bike_id}] Barrier detected — stopping')
            bike.vehicle.set_autopilot(False, tm.get_port())
            bike.vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, brake=1.0))
            bike.stop_timer  = 0.0
            bike.last_detect = 0.0
            bike.state       = S_STOPPED
            ros_node.pub_status(bike.bike_id, S_STOPPED)
            return

        # ── Advance path ──────────────────────────────────────
        # Pop waypoints the vehicle has already passed
        while (bike.path and
               dist2d(veh_loc, bike.path[0].transform.location) < 4.0):
            bike.path.pop(0)

        if bike.path:
            locs = [wp.transform.location
                    for wp in bike.path[:PATH_LOOKAHEAD]]
            tm.set_path(bike.vehicle, locs)
            ros_node.pub_path(bike.bike_id, bike.path)
        else:
            # Path exhausted without reaching goal — pick new goal
            assign_new_goal(bike, tm, goals, ros_node)

    elif bike.state == S_STOPPED:
        # Keep braking — TM must be off or it will override
        bike.vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0))
        bike.stop_timer += TICK_DT

        # Publish detection event periodically while stopped
        now = time.time()
        if barriers and (now - bike.last_detect) >= DETECT_INTERVAL:
            with bike._gps_lock:
                lat = bike.lat
                lon = bike.lon
            ros_node.pub_detection(bike.bike_id, lat, lon, len(barriers))
            bike.last_detect = now

        if bike.stop_timer >= STOP_DURATION:
            bike.state = S_REROUTE
            ros_node.pub_status(bike.bike_id, S_REROUTE)

    elif bike.state == S_REROUTE:
        alt = reroute_around(
            world.get_map(), veh_loc, bike.road_goal, barriers)

        if alt:
            bike.path = alt
            locs = [wp.transform.location for wp in alt[:PATH_LOOKAHEAD]]
            bike.vehicle.set_autopilot(True, tm.get_port())
            tm.set_path(bike.vehicle, locs)
            tm.set_desired_speed(bike.vehicle, SPEED_NORMAL)
            ros_node.pub_path(bike.bike_id, bike.path)
            logging.info(f'[{bike.bike_id}] Rerouted via {len(alt)} waypoints')
            bike.state = S_NAV
            ros_node.pub_status(bike.bike_id, S_NAV)
        else:
            # No reroute found — U-turn
            logging.info(f'[{bike.bike_id}] No reroute — U-turn')
            bike.vehicle.set_autopilot(False, tm.get_port())
            bike.vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.4, brake=0.0,
                    steer=0.5,    reverse=True))
            bike.uturn_timer = 0.0
            bike.state       = S_UTURN
            ros_node.pub_status(bike.bike_id, S_REROUTE)

    elif bike.state == S_UTURN:
        bike.vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.4, brake=0.0,
                steer=0.5,    reverse=True))
        bike.uturn_timer += TICK_DT
        if bike.uturn_timer >= UTURN_REVERSE_TIME:
            # Brief brake before handing back to TM
            bike.vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, brake=1.0))
            time.sleep(0.2)
            assign_new_goal(bike, tm, goals, ros_node)

    elif bike.state == S_WAIT:
        bike.vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=0.8))
        bike.wait_timer += TICK_DT
        if bike.wait_timer >= WAIT_AT_GOAL:
            assign_new_goal(bike, tm, goals, ros_node)


def make_gnss_callback(bike):
    """Return a GNSS callback that writes lat/lon into bike.lat/lon."""
    def cb(msg: NavSatFix):
        with bike._gps_lock:
            bike.lat = msg.latitude
            bike.lon = msg.longitude
    return cb


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='VisionFleet — Bike Navigator Node')
    parser.add_argument('--host',    default='localhost')
    parser.add_argument('--port',    default=2000, type=int)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO)

    # ── CARLA connection ──────────────────────────────────────
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world  = client.get_world()
    cmap   = world.get_map()
    tm     = client.get_trafficmanager()

    # Synchronous mode must already be set by fleet_spawner.py
    # We just ensure TM is in sync mode too
    tm.set_synchronous_mode(True)

    # ── Discover spawned bikes ────────────────────────────────
    logging.info('Discovering bikes from fleet_spawner...')
    bike_pairs = discover_bikes(world)

    if not bike_pairs:
        logging.error(
            'No bikes found. Run fleet_spawner.py first.')
        return

    logging.info(f'Found {len(bike_pairs)} bike(s): '
                 f'{[b for b, _ in bike_pairs]}')

    # ── ROS 2 ─────────────────────────────────────────────────
    rclpy.init()
    ros_node = BikeNavigatorNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(ros_node)

    # ROS spin in background — never competes with tick loop
    spin_thread = threading.Thread(
        target=executor.spin, daemon=True)
    spin_thread.start()

    # ── Build BikeNav objects ─────────────────────────────────
    palette = [
        (0,   255, 136), (0,   180, 216), (244, 162,  97),
        (255,  80, 160), (160, 100, 255), (80,  220, 255),
        (255, 220,  50),
    ]
    bikes = []
    for i, (bike_id, vehicle) in enumerate(bike_pairs):
        color = palette[i % len(palette)]
        bike  = BikeNav(bike_id, vehicle, color)

        ros_node.ensure_publishers(bike_id)
        ros_node.ensure_gnss_sub(bike_id, make_gnss_callback(bike))

        # TM configuration per bike
        tm.auto_lane_change(vehicle, False)
        tm.ignore_lights_percentage(vehicle, 80)
        tm.set_desired_speed(vehicle, SPEED_NORMAL)

        bikes.append(bike)
        logging.info(f'[{bike_id}] Ready')

    # ── Build goal pool ───────────────────────────────────────
    goals = find_delivery_goals(cmap, count=60)
    logging.info(f'Goal pool: {len(goals)} delivery locations')

    if not goals:
        logging.error('No delivery goals found on map')
        rclpy.shutdown()
        return

    # Assign initial goals to all bikes
    for bike in bikes:
        assign_new_goal(bike, tm, goals, ros_node)

    # ── Main tick loop ────────────────────────────────────────
    logging.info('Navigator running. Ctrl+C to stop.')
    last_rediscover = time.time()

    try:
        while True:
            world.tick()

            # Refresh barrier list once per tick (cheap — actors are cached)
            barriers = get_barrier_locations(world)

            # Advance every bike's state machine
            for bike in bikes:
                try:
                    tick_bike(bike, world, tm, goals, ros_node, barriers)
                except Exception as e:
                    logging.warning(
                        f'[{bike.bike_id}] tick error: {e}')

            # Periodically check for newly spawned bikes
            now = time.time()
            if now - last_rediscover > REDISCOVER_INTERVAL:
                last_rediscover = now
                new_pairs = discover_bikes(world)
                known_ids = {b.bike_id for b in bikes}
                for bid, vehicle in new_pairs:
                    if bid not in known_ids:
                        color = palette[len(bikes) % len(palette)]
                        bike  = BikeNav(bid, vehicle, color)
                        ros_node.ensure_publishers(bid)
                        ros_node.ensure_gnss_sub(
                            bid, make_gnss_callback(bike))
                        tm.auto_lane_change(vehicle, False)
                        tm.ignore_lights_percentage(vehicle, 80)
                        tm.set_desired_speed(vehicle, SPEED_NORMAL)
                        assign_new_goal(bike, tm, goals, ros_node)
                        bikes.append(bike)
                        logging.info(f'[{bid}] Discovered and added')

    except KeyboardInterrupt:
        logging.info('Stopping navigator...')
    finally:
        # Turn off autopilot cleanly
        for bike in bikes:
            try:
                bike.vehicle.set_autopilot(False, tm.get_port())
            except Exception:
                pass
        executor.shutdown(wait=False)
        try:
            rclpy.shutdown()
        except Exception:
            pass
        logging.info('Navigator stopped.')


if __name__ == '__main__':
    main()