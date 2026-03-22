#!/usr/bin/env python3
"""
map_node.py — VisionFleet
Collaborative Spatial Consensus Engine with Quorum Sensing.
Implements multi-agent validation and localized centroid clustering.
Uses Hysteresis/Exit-Validation to penalize only AFTER traversing a zone.
"""
import json
import math
import os
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix

ZONES_JSON        = os.path.expanduser('~/VisionFleet_ws/assets/active_zones.json')

# ── Academic Tuning Parameters ────────────────────────────────
ZONE_RADIUS_M     = 15.0    # Radius for clustering and transit tracking
CONF_INITIAL      = 0.40   # Starting confidence when first detected
CONF_HIT_BOOST    = 0.15   # Gentle boost per frame
CONF_MISS_PENALTY = 0.35   # Penalty applied upon EXITING without seeing it
CONF_REMOVE       = 0.15   # Deletion threshold

# Quorum Constraints
MAX_CONF_SINGLE_AGENT = 0.65  
MAX_CONF_MULTI_AGENT  = 1.00  

R_EARTH = 6371000.0 

class EventZone:
    def __init__(self, zone_id, lat, lon, bike_id):
        self.id           = zone_id
        self.lat          = lat
        self.lon          = lon
        self.confidence   = CONF_INITIAL
        self.report_count = 1
        self.reporter_bike = bike_id
        
        self.unique_observers = {bike_id}
        
        # 'inside' tracks if the bike is currently inside the radius
        # 'saw_it_this_pass' tracks if they generated a hit during the current transit
        self.bike_states = {bike_id: {
            'last_hit': time.time(), 
            'inside': True, 
            'saw_it_this_pass': True
        }}

    def distance_to_meters(self, target_lat, target_lon):
        phi1, phi2 = math.radians(self.lat), math.radians(target_lat)
        dphi       = math.radians(target_lat - self.lat)
        dlambda    = math.radians(target_lon - self.lon)
        
        a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R_EARTH * c

    def register_hit(self, lat, lon, bike_id):
        now = time.time()
        
        # Dynamic Centroid
        w = self.report_count
        self.lat = (self.lat * w + lat) / (w + 1)
        self.lon = (self.lon * w + lon) / (w + 1)
        self.report_count += 1
        
        self.unique_observers.add(bike_id)  
        
        if bike_id not in self.bike_states:
            self.bike_states[bike_id] = {'last_hit': 0.0, 'inside': True, 'saw_it_this_pass': True}
            
        self.bike_states[bike_id]['last_hit'] = now
        self.bike_states[bike_id]['inside'] = True
        self.bike_states[bike_id]['saw_it_this_pass'] = True

        conf_cap = MAX_CONF_MULTI_AGENT if len(self.unique_observers) >= 2 else MAX_CONF_SINGLE_AGENT
        self.confidence = min(conf_cap, self.confidence + CONF_HIT_BOOST)

    def update_transit_state(self, bike_id, distance):
        """
        Tracks if a bike enters or exits the zone. 
        Returns True ONLY if the bike exits the zone WITHOUT having seen the obstacle.
        """
        if bike_id not in self.bike_states:
            self.bike_states[bike_id] = {'last_hit': 0.0, 'inside': False, 'saw_it_this_pass': False}
            
        state = self.bike_states[bike_id]
        is_currently_inside = (distance < ZONE_RADIUS_M)

        # Bike just entered the zone
        if is_currently_inside and not state['inside']:
            state['inside'] = True
            state['saw_it_this_pass'] = False # Reset transit flag
            return False

        # Bike just EXITED the zone
        if not is_currently_inside and state['inside']:
            state['inside'] = False
            # If they traversed it but didn't see it, trigger penalty
            if not state['saw_it_this_pass']:
                self.confidence = max(0.0, self.confidence - CONF_MISS_PENALTY)
                return True
                
        return False

    def should_remove(self):
        return self.confidence <= CONF_REMOVE

    def to_dict(self):
        return {
            'id':           self.id,
            'lat':          self.lat,
            'lon':          self.lon,
            'confidence':   round(self.confidence, 3),
            'report_count': self.report_count,
            'bike_count':   len(self.unique_observers),
        }


class MapNode(Node):

    def __init__(self):
        super().__init__('map_node')
        self._zones          = {}
        self._next_id        = 0
        self._subscribed_ids = set()
        
        self._bike_poses     = {} 

        self.create_timer(3.0,  self._discover_bikes)
        self.create_timer(0.5,  self._check_spatial_misses) # Run faster to catch precise exits
        self.create_timer(0.5,  self._export_zones) 

        os.makedirs(os.path.dirname(ZONES_JSON), exist_ok=True)
        with open(ZONES_JSON, 'w') as f:
            json.dump([], f)
        self.get_logger().info('Map Node: Spatio-Temporal Consensus Engine Ready.')

    def _discover_bikes(self):
        known_topics = [name for name, _ in self.get_topic_names_and_types()]
        for topic in known_topics:
            if topic.startswith('/visionfleet/') and topic.endswith('/detections'):
                parts = topic.split('/')
                if len(parts) == 4:
                    bike_id = parts[2]
                    self._subscribe_bike(bike_id)

    def _subscribe_bike(self, bike_id):
        if bike_id in self._subscribed_ids:
            return
            
        self.create_subscription(
            String, f'/visionfleet/{bike_id}/detections', self._on_detection, 10)
            
        gnss_topic = f'/carla/hero/{bike_id}/gnss'
        self.create_subscription(
            NavSatFix, gnss_topic, self._make_gnss_cb(bike_id), 10)

        self._subscribed_ids.add(bike_id)
        self.get_logger().info(f'[map] Tracking kinematics/perception for: {bike_id}')

    def _make_gnss_cb(self, bike_id):
        def cb(msg: NavSatFix):
            self._bike_poses[bike_id] = (msg.latitude, msg.longitude, time.time())
        return cb

    def _on_detection(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if not data.get('road_blocked', False):
            return

        lat     = data.get('lat', 0.0)
        lon     = data.get('lon', 0.0)
        bike_id = data.get('bike_id', 'unknown')

        if lat == 0.0 and lon == 0.0:
            return

        nearest, nearest_dist = None, float('inf')
        for zone in self._zones.values():
            d = zone.distance_to_meters(lat, lon)
            if d < nearest_dist:
                nearest_dist = d
                nearest = zone

        if nearest is not None and nearest_dist < ZONE_RADIUS_M:
            nearest.register_hit(lat, lon, bike_id)
            self.get_logger().info(
                f'Zone {nearest.id} REINFORCED by {bike_id} (Conf: {nearest.confidence:.2f}, Observers: {len(nearest.unique_observers)})')
        else:
            zone = EventZone(self._next_id, lat, lon, bike_id)
            self._zones[self._next_id] = zone
            self.get_logger().info(f'NEW anomaly {self._next_id} detected by {bike_id}')
            self._next_id += 1

    def _check_spatial_misses(self):
        now = time.time()
        for bike_id, (lat, lon, last_msg_time) in self._bike_poses.items():
            if now - last_msg_time > 5.0:
                continue
                
            for zone_id, zone in list(self._zones.items()):
                # A bike cannot invalidate a zone it originally reported.
                if bike_id == zone.reporter_bike:
                    continue
                    
                d = zone.distance_to_meters(lat, lon)
                
                # Check if bike completed a transit without seeing the obstacle
                if zone.update_transit_state(bike_id, d):
                    self.get_logger().warning(
                        f'Zone {zone_id} PENALIZED! {bike_id} exited area finding nothing. '
                        f'Conf→{zone.confidence:.2f}')
                        
        dead = [zid for zid, z in self._zones.items() if z.should_remove()]
        for zid in dead:
            self.get_logger().info(f'Zone {zid} REMOVED (Rejected by fleet transit)')
            del self._zones[zid]

    def _export_zones(self):
        zones_list = [z.to_dict() for z in self._zones.values()]
        try:
            with open(ZONES_JSON, 'w') as f:
                json.dump(zones_list, f)
        except Exception as e:
            self.get_logger().warn(f'Zone export failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = MapNode()
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