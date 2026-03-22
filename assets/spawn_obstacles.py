#!/usr/bin/env python3

import os
import json
import carla
import math
import random
import time
import argparse
import signal
import sys

MAX_BLOCKS_LIMIT    = 10
BARRIER_YAW_JITTER  = 16.0
BARRIER_OFFSET_LAT  = 0.4
CONE_DISTANCE_AHEAD = 1.5
CONE_JITTER_FWD     = 0.2
CONE_JITTER_LAT     = 0.15
CONE_YAW_RANDOMNESS = 60.0
BARRIER_Z_OFFSET    = 0.05
CONE_Z_OFFSET       = 0.05

JSON_PATH = os.path.expanduser('~/VisionFleet_ws/assets/active_roadblocks.json')

def get_perpendicular_offset(transform, metres):
    yaw = math.radians(transform.rotation.yaw)
    right_x = math.sin(yaw)
    right_y = -math.cos(yaw)
    loc = transform.location
    return carla.Location(x=loc.x + right_x * metres, y=loc.y + right_y * metres, z=loc.z)

def collect_all_lanes(anchor_wp, carla_map):
    lanes = []
    seen_ids = set()
    
    yaw = math.radians(anchor_wp.transform.rotation.yaw)
    right_x = math.sin(yaw)
    right_y = -math.cos(yaw)
    loc = anchor_wp.transform.location
    
    for step in range(-20, 21):
        offset = step * 1.5
        check_loc = carla.Location(
            x=loc.x + right_x * offset,
            y=loc.y + right_y * offset,
            z=loc.z
        )
        wp = carla_map.get_waypoint(check_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        if wp and wp.road_id == anchor_wp.road_id:
            if wp.lane_id not in seen_ids:
                lanes.append(wp)
                seen_ids.add(wp.lane_id)
                
    return lanes

def spawn_road_block(world, bp_lib, anchor_wp, rng, cmap):
    spawned = []
    
    cone_bps = [b for b in [
        bp_lib.find('static.prop.trafficcone01'),
    ] if b is not None]

    barrier_bp = bp_lib.find('static.prop.streetbarrier')

    if not cone_bps or barrier_bp is None:
        return spawned, None

    lanes = collect_all_lanes(anchor_wp, cmap)
    
    cx = sum(l.transform.location.x for l in lanes) / len(lanes)
    cy = sum(l.transform.location.y for l in lanes) / len(lanes)
    total_width = sum(l.lane_width for l in lanes)
    base_yaw = anchor_wp.transform.rotation.yaw
    
    center_data = {"x": cx, "y": cy, "yaw": base_yaw, "width": total_width}

    for lane_wp in lanes:
        tf = lane_wp.transform
        lw = lane_wp.lane_width
        yaw_rad = math.radians(tf.rotation.yaw)

        barrier_offsets = [-lw * BARRIER_OFFSET_LAT, lw * BARRIER_OFFSET_LAT]
        for offset in barrier_offsets:
            loc = get_perpendicular_offset(tf, offset)
            loc.z += BARRIER_Z_OFFSET
            barrier_yaw = base_yaw + 90 + rng.uniform(-BARRIER_YAW_JITTER, BARRIER_YAW_JITTER)
            try:
                a = world.spawn_actor(barrier_bp, carla.Transform(loc, carla.Rotation(yaw=barrier_yaw)))
                spawned.append(a)
            except Exception:
                pass

        cone_offsets = [-lw * 0.35, 0.0, lw * 0.35]
        for offset in cone_offsets:
            jitter_fwd = rng.uniform(-CONE_JITTER_FWD, CONE_JITTER_FWD)
            jitter_lat = rng.uniform(-CONE_JITTER_LAT, CONE_JITTER_LAT)
            base = get_perpendicular_offset(tf, offset + jitter_lat)
            cone_loc = carla.Location(
                x=base.x + math.cos(yaw_rad) * (CONE_DISTANCE_AHEAD + jitter_fwd),
                y=base.y + math.sin(yaw_rad) * (CONE_DISTANCE_AHEAD + jitter_fwd),
                z=base.z + CONE_Z_OFFSET
            )
            cone_yaw = tf.rotation.yaw + rng.uniform(-CONE_YAW_RANDOMNESS, CONE_YAW_RANDOMNESS)
            try:
                a = world.spawn_actor(rng.choice(cone_bps), carla.Transform(cone_loc, carla.Rotation(yaw=cone_yaw)))
                spawned.append(a)
            except Exception:
                pass

    return spawned, center_data

def remove_all_obstacles(world):
    actors = world.get_actors()
    for a in actors:
        if any(k in a.type_id for k in ['cone', 'barrier']):
            try:
                a.destroy()
            except Exception:
                pass
    
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'w') as f:
            json.dump([], f)

def _cleanup(signum=None, frame=None):
        for a in all_spawned:
            try:
                a.destroy()
            except Exception:
                pass
        with open(JSON_PATH, 'w') as f:
            json.dump([], f)
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove', action='store_true')
    parser.add_argument('-n', '--num', type=int, default=5)
    args = parser.parse_args()

    num_blocks_to_spawn = min(args.num, MAX_BLOCKS_LIMIT)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    if args.remove:
        remove_all_obstacles(world)
        return

    bp_lib = world.get_blueprint_library()
    cmap = world.get_map()
    rng = random.Random()

    spawn_points = cmap.get_spawn_points()
    candidates = []
    seen_roads = set()
    
    for sp in spawn_points:
        wp = cmap.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction or wp.road_id in seen_roads:
            continue
        seen_roads.add(wp.road_id)
        candidates.append(wp)

    rng.shuffle(candidates)
    selected = candidates[:num_blocks_to_spawn]

    all_spawned = []
    roadblocks_data = []
    
    for wp in selected:
        actors, center_data = spawn_road_block(world, bp_lib, wp, rng, cmap)
        all_spawned.extend(actors)
        if center_data:
            roadblocks_data.append(center_data)

    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, 'w') as f:
        json.dump(roadblocks_data, f, indent=4)

    def _cleanup(signum=None, frame=None):
        for a in all_spawned:
            try:
                a.destroy()
            except Exception:
                pass
        with open(JSON_PATH, 'w') as f:
            json.dump([], f)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT,  _cleanup)

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        _cleanup()

if __name__ == '__main__':
    main()