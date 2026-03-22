from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='visionfleet_biker',
            executable='carla_observer_node.py',
            name='carla_observer_node',
            output='screen'
        ),
        Node(
            package='visionfleet_biker',
            executable='fusion_node.py',
            name='fusion_node',
            output='screen'
        ),
        Node(
            package='visionfleet_biker',
            executable='map_node.py',
            name='map_node',
            output='screen'
        ),
        Node(
            package='visionfleet_biker',
            executable='gps_map_node.py',
            name='gps_map_node',
            output='screen'
        ),
    ])