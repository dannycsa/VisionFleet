#!/usr/bin/env python3
"""
carla_observer_node.py — VisionFleet
Polls CARLA for world-level data (obstacles, spectator) and publishes
them as proper ROS 2 topics. No UI, no bike logic.
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseArray, Pose
from std_msgs.msg import Float32MultiArray
import carla


class CarlaObserverNode(Node):

    def __init__(self):
        super().__init__('carla_observer_node')

        self._client = None
        self._world  = None
        self._map    = None
        self._connect()

        # Publishers
        self._pub_spectator = self.create_publisher(
            PointStamped, '/visionfleet/world/spectator', 10)
        self._pub_obstacles = self.create_publisher(
            Float32MultiArray, '/visionfleet/world/obstacles', 10)

        # Poll at 2 Hz — CARLA world data doesn't change faster than this
        self.create_timer(0.5, self._publish_spectator)
        self.create_timer(1.0, self._publish_obstacles)

        self.get_logger().info('CARLA observer node ready')

    def _connect(self):
        try:
            self._client = carla.Client('localhost', 2000)
            self._client.set_timeout(4.0)
            self._world  = self._client.get_world()
            self._map    = self._world.get_map()
            self.get_logger().info('Connected to CARLA')
        except Exception as e:
            self.get_logger().warn(f'CARLA not available yet: {e}')

    def _ensure(self):
        if self._world is None:
            self._connect()
        return self._world is not None

    def _publish_spectator(self):
        if not self._ensure():
            return
        try:
            loc = self._world.get_spectator().get_transform().location
            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            msg.point.x = loc.x
            msg.point.y = loc.y
            msg.point.z = loc.z
            self._pub_spectator.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Spectator error: {e}')

    def _publish_obstacles(self):
        if not self._ensure():
            return
        try:
            # Each obstacle: x, y, yaw, lane_width  (4 floats per entry)
            data = []
            for a in self._world.get_actors():
                if 'streetbarrier' in a.type_id:
                    loc = a.get_transform().location
                    yaw = a.get_transform().rotation.yaw
                    wp  = self._map.get_waypoint(loc)
                    lw  = wp.lane_width if wp else 3.5
                    data.extend([loc.x, loc.y, yaw, lw])
            msg = Float32MultiArray()
            msg.data = data
            self._pub_obstacles.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Obstacles error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = CarlaObserverNode()
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