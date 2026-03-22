#!/usr/bin/env python3
"""
fusion_node.py — VisionFleet
Discovers bikes dynamically by polling for /carla/hero/*/gnss topics.
Converts GNSS to world XY, computes yaw, publishes fused_pose per bike.
No hardcoded bike list — works for any quantity spawned at runtime.
No CARLA dependency — pure ROS 2.
"""
import math
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PointStamped

EARTH_R = 6378137.0


def gnss_to_xy(lat, lon):
    x = lon * (EARTH_R * math.pi / 180.0)
    y = lat * (EARTH_R * math.pi / 180.0)
    return x, y


class BikeTracker:
    def __init__(self, bike_id, publisher):
        self.bike_id   = bike_id
        self.publisher = publisher
        self.prev_x    = None
        self.prev_y    = None
        self.yaw       = 0.0

    def update(self, x, y, stamp):
        if self.prev_x is not None:
            dx = x - self.prev_x
            dy = y - self.prev_y
            if abs(dx) + abs(dy) > 0.05:
                self.yaw = math.degrees(math.atan2(dy, dx))
        self.prev_x = x
        self.prev_y = y

        msg = PointStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = self.bike_id
        msg.point.x = x
        msg.point.y = y
        msg.point.z = self.yaw  # yaw packed into z — avoids custom msg
        self.publisher.publish(msg)


class FusionNode(Node):

    def __init__(self):
        super().__init__('fusion_node')
        self._lock  = threading.Lock()
        self._bikes = {}  # bike_id -> BikeTracker

        # Poll ROS topics every 2s to discover newly spawned bikes
        self.create_timer(2.0, self._discover_bikes)
        self.get_logger().info('Fusion node ready — watching for bikes...')

    def _discover_bikes(self):
        """
        Scan active topics for GNSS patterns:
          /carla/hero/gnss              → single hero bike
          /carla/hero/<bike_id>/gnss    → fleet bikes (bike_1, bike_2, ...)
        """
        known_topics = [name for name, _ in self.get_topic_names_and_types()]

        for topic in known_topics:
            # Single hero bike
            if topic == '/carla/hero/gnss':
                with self._lock:
                    self._register_bike('hero', '/carla/hero/gnss')

            # Fleet bikes: /carla/hero/<bike_id>/gnss
            elif topic.startswith('/carla/hero/') and topic.endswith('/gnss'):
                parts = topic.split('/')
                # ['', 'carla', 'hero', '<bike_id>', 'gnss']
                if len(parts) == 5:
                    bike_id = parts[3]
                    with self._lock:
                        self._register_bike(bike_id, topic)

    def _register_bike(self, bike_id, gnss_topic):
        """Safe to call multiple times — only registers once per bike."""
        if bike_id in self._bikes:
            return

        pub = self.create_publisher(
            PointStamped, f'/visionfleet/{bike_id}/fused_pose', 10)

        self._bikes[bike_id] = BikeTracker(bike_id, pub)

        self.create_subscription(
            NavSatFix, gnss_topic,
            self._make_gnss_cb(bike_id), 10)

        self.get_logger().info(
            f'[fusion] Registered: {bike_id}  ←  {gnss_topic}')

    def _make_gnss_cb(self, bike_id):
        def cb(msg):
            x, y = gnss_to_xy(msg.latitude, msg.longitude)
            with self._lock:
                if bike_id in self._bikes:
                    self._bikes[bike_id].update(x, y, msg.header.stamp)
        return cb


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
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