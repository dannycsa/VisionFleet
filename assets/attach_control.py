#!/usr/bin/env python3
"""
attach_control.py — VisionFleet
Attach keyboard control to any bike spawned by fleet_spawner.py.

Usage:
  python3 attach_control.py --bike hero     # the original single bike
  python3 attach_control.py --bike bike_1   # first fleet bike
  python3 attach_control.py --bike bike_2   # second fleet bike
"""
import carla
import pygame
from pygame.locals import (K_UP, K_DOWN, K_LEFT, K_RIGHT,
                            K_w, K_a, K_s, K_d,
                            K_SPACE, K_ESCAPE, K_q,
                            K_m, K_COMMA, K_PERIOD, K_p)
import sys
import argparse


def find_vehicle_for_bike(world, bike_id):
    """
    All fleet bikes share role_name='hero' at the vehicle level.
    We identify which vehicle belongs to bike_id by checking
    whether any attached sensor has ros_name starting with '<bike_id>/'.

    For the plain 'hero' bike (no fleet prefix), sensors have
    ros_name like 'rgb', 'gnss' — no slash prefix.
    """
    all_actors = world.get_actors()

    for actor in all_actors.filter('vehicle.*'):
        role = actor.attributes.get('role_name', '')
        if role != 'hero':
            continue

        if bike_id == 'hero':
            # Hero bike sensors have simple names: 'rgb', 'gnss', 'imu'
            # They do NOT have a slash in their ros_name
            attached = [
                a for a in all_actors
                if hasattr(a, 'parent') and a.parent is not None
                and a.parent.id == actor.id
            ]
            sensor_names = [
                a.attributes.get('ros_name', '') for a in attached
            ]
            # Hero sensors: no slash. Fleet sensors: 'bike_N/something'
            if any('/' not in n and n != '' for n in sensor_names):
                return actor

        else:
            # Fleet bike: look for a sensor with ros_name 'bike_N/gnss'
            attached = [
                a for a in all_actors
                if hasattr(a, 'parent') and a.parent is not None
                and a.parent.id == actor.id
            ]
            sensor_names = [
                a.attributes.get('ros_name', '') for a in attached
            ]
            if any(n.startswith(f'{bike_id}/') for n in sensor_names):
                return actor

    return None


class KeyboardControl:
    def __init__(self, vehicle, bike_id):
        self._vehicle           = vehicle
        self._control           = carla.VehicleControl()
        self._steer_cache       = 0.0
        self._autopilot_enabled = False
        self._vehicle.set_autopilot(False)
        print(f"Attached to [{bike_id}]. Manual control active.")
        print("  WASD / arrows = drive   |   P = toggle autopilot   |   Q/ESC = quit")

    def parse_events(self, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key in (K_ESCAPE, K_q):
                    return True
                elif event.key == K_m:
                    self._control.manual_gear_shift = \
                        not self._control.manual_gear_shift
                    self._control.gear = \
                        self._vehicle.get_control().gear
                    print(f"Manual gear: {self._control.manual_gear_shift}")
                elif self._control.manual_gear_shift and event.key == K_COMMA:
                    self._control.gear = max(-1, self._control.gear - 1)
                elif self._control.manual_gear_shift and event.key == K_PERIOD:
                    self._control.gear = self._control.gear + 1
                elif event.key == K_p:
                    self._autopilot_enabled = not self._autopilot_enabled
                    self._vehicle.set_autopilot(self._autopilot_enabled)
                    print(f"Autopilot: {'ON' if self._autopilot_enabled else 'OFF'}")

        if not self._autopilot_enabled:
            self._parse_keys(pygame.key.get_pressed(), clock.get_time())
            self._control.reverse = self._control.gear < 0
            self._vehicle.apply_control(self._control)

        return False

    def _parse_keys(self, keys, ms):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.1, 1.0)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1.0)
        else:
            self._control.brake = 0.0

        inc = 5e-4 * ms
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache = 0.0 if self._steer_cache > 0 \
                                else self._steer_cache - inc
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache = 0.0 if self._steer_cache < 0 \
                                else self._steer_cache + inc
        else:
            self._steer_cache = 0.0

        self._steer_cache       = max(-0.7, min(0.7, self._steer_cache))
        self._control.steer     = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]


def main():
    argparser = argparse.ArgumentParser(description='VisionFleet — Attach Control')
    argparser.add_argument('--host', default='127.0.0.1')
    argparser.add_argument('-p', '--port', default=2000, type=int)
    argparser.add_argument('--bike', default='hero',
                           help='Bike ID to control: hero, bike_1, bike_2, ... '
                                '(default: hero)')
    args = argparser.parse_args()

    pygame.init()
    display = pygame.display.set_mode((340, 220))
    pygame.display.set_caption(f'VisionFleet — controlling [{args.bike}]')
    display.fill((30, 30, 40))
    font  = pygame.font.SysFont('monospace', 13)
    lines = [
        f'Controlling: {args.bike}',
        '',
        'WASD / arrows = drive',
        'P             = toggle autopilot',
        'Q / ESC       = quit',
    ]
    for i, line in enumerate(lines):
        display.blit(font.render(line, True, (200, 200, 210)), (20, 20 + i * 22))
    pygame.display.flip()

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()

    print(f'Looking for bike: [{args.bike}]...')
    vehicle = find_vehicle_for_bike(world, args.bike)

    if vehicle is None:
        print(f"Error: could not find bike '{args.bike}' in the world.")
        print("Make sure fleet_spawner.py is running and the bike ID is correct.")
        print("Available bikes are named: hero, bike_1, bike_2, ...")
        pygame.quit()
        sys.exit(1)

    print(f"Found vehicle id={vehicle.id} for bike '{args.bike}'")
    controller = KeyboardControl(vehicle, args.bike)
    clock      = pygame.time.Clock()

    try:
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(clock):
                break
    finally:
        print(f"\nRestoring autopilot for [{args.bike}] and detaching.")
        try:
            target_vehicle.set_autopilot(True)
        except Exception:
            pass
        pygame.quit()


if __name__ == '__main__':
    main()