#!/usr/bin/env python3
"""
fleet_driver.py — VisionFleet
Reads spawned_bikes.json written by fleet_spawner.py,
finds the vehicles in CARLA, starts autopilot, and runs
the simulation tick loop.

This is the file to modify when replacing autopilot with
a custom path planner node.
"""
import argparse
import json
import logging
import os
import time
import carla

SPAWNED_JSON = os.path.join(os.path.dirname(__file__), "spawned_bikes.json")


def find_vehicle(world, actor_id):
    """Find a CARLA actor by ID. Returns None if not found."""
    for actor in world.get_actors():
        if actor.id == actor_id:
            return actor
    return None


def main(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world  = client.get_world()

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    # Load registry written by fleet_spawner.py
    if not os.path.exists(SPAWNED_JSON):
        logging.error(f"Registry not found: {SPAWNED_JSON}")
        logging.error("Run fleet_spawner.py first.")
        return

    with open(SPAWNED_JSON) as f:
        spawned_info = json.load(f)

    if not spawned_info:
        logging.error("Registry is empty — no bikes to drive.")
        return

    # Resolve actor IDs to live CARLA actors
    vehicles = []
    for entry in spawned_info:
        bike_name = entry["bike_name"]
        actor_id  = entry["actor_id"]
        vehicle   = find_vehicle(world, actor_id)
        if vehicle is None:
            logging.warning(f"[{bike_name}] actor {actor_id} not found — skipped")
            continue
        vehicles.append((bike_name, vehicle))
        logging.info(f"Found [{bike_name}] actor_id={actor_id}")

    if not vehicles:
        logging.error("No live vehicles found. Did fleet_spawner.py run recently?")
        return

    # ── Start movement ────────────────────────────────────────
    # REPLACE THIS BLOCK when switching to a path planner.
    # For now: autopilot via CARLA traffic manager.
    for bike_name, vehicle in vehicles:
        vehicle.set_autopilot(True)
        logging.info(f"[{bike_name}] autopilot ON")

    logging.info(f"Driving {len(vehicles)} bike(s). Ctrl+C to stop.")

    try:
        while True:
            world.tick()
    except KeyboardInterrupt:
        logging.info("Stopping...")
    finally:
        for bike_name, vehicle in vehicles:
            try:
                vehicle.set_autopilot(False)
                logging.info(f"[{bike_name}] autopilot OFF")
            except Exception:
                pass


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='VisionFleet — Fleet Driver')
    argparser.add_argument('--host', default='localhost')
    argparser.add_argument('--port', default=2000, type=int)
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=logging.DEBUG if args.debug else logging.INFO)
    logging.info('Connecting to %s:%s', args.host, args.port)
    main(args)