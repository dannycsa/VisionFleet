import carla, random, time
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world  = client.get_world()
cmap   = world.get_map()
tm     = client.get_trafficmanager()

s = world.get_settings()
s.synchronous_mode = True
s.fixed_delta_seconds = 0.05
world.apply_settings(s)
tm.set_synchronous_mode(True)

bp = world.get_blueprint_library().filter('vehicle.yamaha.yzf')[0]
bp.set_attribute('role_name', 'test')
sp = cmap.get_spawn_points()[0]
v  = world.spawn_actor(bp, sp)
world.tick()

v.set_autopilot(True)
tm.set_desired_speed(v, 20.0)

# Simple straight path - 50 waypoints ahead
wp = cmap.get_waypoint(sp.location, project_to_road=True)
path = []
for _ in range(50):
    nexts = wp.next(2.0)
    if not nexts: break
    wp = nexts[0]
    path.append(wp.transform.location)

tm.set_path(v, path)
print(f"Path set with {len(path)} points — watching for 10s")

for _ in range(200):
    world.tick()
    loc = v.get_transform().location
    print(f"  pos: {loc.x:.1f}, {loc.y:.1f}")
    time.sleep(0.05)

v.destroy()
world.apply_settings(world.get_settings())