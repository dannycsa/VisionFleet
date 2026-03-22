import carla
import cv2
import numpy as np

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
cmap = world.get_map()

# 1. Generate high-density waypoints
waypoints = cmap.generate_waypoints(0.5)

# Get dynamic map boundaries to calculate image size
x_coords = [wp.transform.location.x for wp in waypoints]
y_coords = [wp.transform.location.y for wp in waypoints]

min_x, max_x = min(x_coords) - 10, max(x_coords) + 10
min_y, max_y = min(y_coords) - 10, max(y_coords) + 10

PIXELS_PER_METER = 20  # Resolution scale
width = int((max_x - min_x) * PIXELS_PER_METER)
height = int((max_y - min_y) * PIXELS_PER_METER)

# 2. Define colors (BGR format for OpenCV)
BG_COLOR = (107, 107, 107)       # Dark gray background (#6b6b6b)
ROAD_COLOR = (50, 50, 50)        # Dark asphalt
SIDEWALK_COLOR = (130, 130, 130) # Lighter sidewalks
LANE_LINE_COLOR = (200, 200, 200)

# Create empty matrices (Binary masks)
road_mask = np.zeros((height, width), dtype=np.uint8)
sidewalk_mask = np.zeros((height, width), dtype=np.uint8)

def world_to_pixel(x, y):
    u = int((x - min_x) * PIXELS_PER_METER)
    v = int((y - min_y) * PIXELS_PER_METER)
    return (u, v)

# 3. Draw asphalt and sidewalk areas onto the masks
for wp in waypoints:
    u, v = world_to_pixel(wp.transform.location.x, wp.transform.location.y)
    # Radius based on the actual lane width provided by the CARLA API
    radius = int((wp.lane_width / 2.0) * PIXELS_PER_METER)

    if wp.lane_type == carla.LaneType.Driving:
        cv2.circle(road_mask, (u, v), radius, 255, -1)
    elif wp.lane_type == carla.LaneType.Sidewalk:
        cv2.circle(sidewalk_mask, (u, v), int(radius * 0.9), 255, -1)

# 4. Morphological Operations (Computer Vision)
# Fills gaps and creates perfect, continuous road edges
kernel_size = int(PIXELS_PER_METER * 1.5)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
sidewalk_mask = cv2.morphologyEx(sidewalk_mask, cv2.MORPH_CLOSE, kernel)

# 5. Final Image Composition
final_map = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)

# Apply the layers with their respective colors
final_map[sidewalk_mask == 255] = SIDEWALK_COLOR
final_map[road_mask == 255] = ROAD_COLOR

# 6. Add lane details (Center markings)
lane_waypoints = cmap.generate_waypoints(2.0)
for wp in lane_waypoints:
    if wp.lane_type == carla.LaneType.Driving:
        u, v = world_to_pixel(wp.transform.location.x, wp.transform.location.y)
        cv2.circle(final_map, (u, v), 1, LANE_LINE_COLOR, -1)

# 7. Save the final map
output_path = '/home/danny/VisionFleet_ws/assets/town10_map.png'
cv2.imwrite(output_path, final_map)
print(f"Saved at: {output_path}")

# Save bounds to a file so gps_map_node can read them
import json
bounds = {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y}
with open('/home/danny/VisionFleet_ws/assets/town10_bounds.json', 'w') as f:
    json.dump(bounds, f)
print(f"Bounds: {bounds}")