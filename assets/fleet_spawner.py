#!/usr/bin/env python3
import argparse
import json
import logging
import math
import os
import copy
import time
import carla

INSTANCES_DIR = os.path.join(os.path.dirname(__file__), "bike_instances")

# ── Barrier detection tuning ──────────────────────────────────
BARRIER_DETECT_DIST = 6.0   # metres — how far ahead to check
STOP_WAIT_TIME      = 4.0   # seconds stopped before warping
SPEED_NORMAL        = 60.0  # km/h handed to traffic manager
TICK_DT             = 0.05  # seconds per world tick


def _generate_bike_configs(template_config, quantity):
    os.makedirs(INSTANCES_DIR, exist_ok=True)
    configs = []
    for i in range(quantity):
        bike_name = f"bike_{i+1}"
        config    = copy.deepcopy(template_config)
        config["id"] = "hero"
        for sensor in config.get("sensors", []):
            sensor["id"] = f"{bike_name}/{sensor['id']}"
        filepath = os.path.join(INSTANCES_DIR, f"{bike_name}.json")
        with open(filepath, "w") as f:
            json.dump(config, f, indent=4)
        configs.append((bike_name, config))
    return configs


def _setup_vehicle(world, config, spawn_index):
    bp_library   = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if spawn_index >= len(spawn_points):
        raise RuntimeError(f"Not enough spawn points for index {spawn_index}")
    bp = bp_library.filter(config.get("type"))[0]
    bp.set_attribute("role_name", config.get("id"))
    bp.set_attribute("ros_name",  config.get("id"))
    return world.spawn_actor(bp, spawn_points[spawn_index], attach_to=None)


def _setup_sensors(world, vehicle, sensors_config):
    bp_library = world.get_blueprint_library()
    sensors    = []
    for sensor in sensors_config:
        bp = bp_library.filter(sensor.get("type"))[0]
        bp.set_attribute("ros_name",  sensor.get("id"))
        bp.set_attribute("role_name", sensor.get("id"))
        for key, value in sensor.get("attributes", {}).items():
            bp.set_attribute(str(key), str(value))
        wp = carla.Transform(
            location=carla.Location(
                x= sensor["spawn_point"]["x"],
                y=-sensor["spawn_point"]["y"],
                z= sensor["spawn_point"]["z"]),
            rotation=carla.Rotation(
                roll= sensor["spawn_point"]["roll"],
                pitch=-sensor["spawn_point"]["pitch"],
                yaw=  -sensor["spawn_point"]["yaw"]))
        sensors.append(world.spawn_actor(bp, wp, attach_to=vehicle))
        sensors[-1].enable_for_ros()
    return sensors


def _get_barrier_locations(world):
    return [a.get_transform().location
            for a in world.get_actors()
            if 'streetbarrier' in a.type_id]


def _barrier_ahead(vehicle_tf, barriers, detect_dist):
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
        if dx * fwd_x + dy * fwd_y > 0:
            return True
    return False


class _BikeState:
    def __init__(self, name, vehicle, tm, cmap):
        self.name         = name
        self.vehicle      = vehicle
        self.tm           = tm
        self.cmap         = cmap
        self.state        = 'roaming'
        self.stop_timer   = 0.0

    def start_roaming(self):
        self.vehicle.set_autopilot(True, self.tm.get_port())
        self.tm.set_desired_speed(self.vehicle, SPEED_NORMAL)
        self.tm.auto_lane_change(self.vehicle, True)
        self.tm.ignore_lights_percentage(self.vehicle, 100)
        self.tm.ignore_signs_percentage(self.vehicle, 100)
        self.state = 'roaming'
        logging.info(f'[{self.name}] roaming')

    def _do_uturn(self):
        tf  = self.vehicle.get_transform()
        loc = tf.location
        yaw = tf.rotation.yaw

        target_wp = None
        
        # Calculate vectors pointing strictly Left and Right of the bike
        left_x = math.cos(math.radians(yaw - 90))
        left_y = math.sin(math.radians(yaw - 90))
        right_x = math.cos(math.radians(yaw + 90))
        right_y = math.sin(math.radians(yaw + 90))

        # Send a geometric probe left and right, step by step, up to 40 meters away
        for offset in range(3, 40, 2):
            for dir_x, dir_y in [(left_x, left_y), (right_x, right_y)]:
                probe_loc = carla.Location(x=loc.x + dir_x * offset, y=loc.y + dir_y * offset, z=loc.z)
                probe_wp = self.cmap.get_waypoint(probe_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                
                if probe_wp:
                    # Make sure the point we found is actually near the probe (didn't just snap back to our lane)
                    if probe_wp.transform.location.distance(probe_loc) < 3.0:
                        wp_yaw = probe_wp.transform.rotation.yaw
                        # Check if it's facing the opposite direction (> 130 degrees difference)
                        yaw_diff = abs(((wp_yaw - yaw + 180) % 360) - 180)
                        if yaw_diff > 130:
                            target_wp = probe_wp
                            break
            if target_wp:
                break

        if target_wp:
            # Move 8 meters ahead in the new direction to securely clear the barrier
            next_wps = target_wp.next(8.0)
            if next_wps:
                target_wp = next_wps[0]

            new_tf = carla.Transform(
                carla.Location(
                    x=target_wp.transform.location.x,
                    y=target_wp.transform.location.y,
                    z=target_wp.transform.location.z + 0.5), # Elevated slightly to prevent clipping
                target_wp.transform.rotation)
            logging.info(f'[{self.name}] warped across median to opposite lane')
        else:
            # Fallback for one-way streets/tight alleys
            new_tf = carla.Transform(
                carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.5),
                carla.Rotation(pitch=tf.rotation.pitch, yaw=yaw + 180.0, roll=tf.rotation.roll))
            logging.warning(f'[{self.name}] No opposite lane found. Spinning 180 in place.')

        self.vehicle.set_autopilot(False, self.tm.get_port())
        self.vehicle.set_transform(new_tf)
        
        time.sleep(0.1) 
        self.start_roaming()

    def tick(self, barriers):
        if self.state == 'roaming':
            tf = self.vehicle.get_transform()
            if barriers and _barrier_ahead(tf, barriers, BARRIER_DETECT_DIST):
                self.vehicle.set_autopilot(False, self.tm.get_port())
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                self.stop_timer = 0.0
                self.state      = 'stopped'
                logging.info(f'[{self.name}] barrier ahead — stopping')

        elif self.state == 'stopped':
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            self.stop_timer += TICK_DT
            if self.stop_timer >= STOP_WAIT_TIME:
                self.state = 'uturn'
                logging.info(f'[{self.name}] executing U-turn')

        elif self.state == 'uturn':
            self._do_uturn()


def main(args):
    world             = None
    vehicles          = []
    sensors           = []
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world  = client.get_world()
        
        cmap = world.get_map()

        original_settings            = world.get_settings()
        settings                     = world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = TICK_DT
        world.apply_settings(settings)

        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)

        with open(args.file) as f:
            template_config = json.load(f)

        bike_configs = _generate_bike_configs(template_config, args.quantity)

        for spawn_index, (bike_name, config) in enumerate(bike_configs):
            vehicle = _setup_vehicle(world, config, spawn_index)
            vehicles.append((bike_name, vehicle))
            bike_sensors = _setup_sensors(world, vehicle, config.get("sensors", []))
            sensors.extend(bike_sensors)

        world.tick()

        bike_states = []
        for bike_name, vehicle in vehicles:
            bs = _BikeState(bike_name, vehicle, tm, cmap)
            bs.start_roaming()
            bike_states.append(bs)

        logging.info(f"Running {len(bike_states)} bike(s) — Ctrl+C to stop.")

        while True:
            world.tick()
            barriers = _get_barrier_locations(world)
            for bs in bike_states:
                bs.tick(barriers)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    finally:
        if original_settings:
            world.apply_settings(original_settings)
        for sensor in sensors:
            try: sensor.destroy()
            except Exception: pass
        for _, vehicle in vehicles:
            try:
                vehicle.set_autopilot(False)
                vehicle.destroy()
            except Exception: pass

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='VisionFleet Spawner')
    argparser.add_argument('--host', default='localhost')
    argparser.add_argument('--port', default=2000, type=int)
    argparser.add_argument('-f', '--file', required=True)
    argparser.add_argument('-q', '--quantity', default=1, type=int)
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG if args.debug else logging.INFO)
    main(args)