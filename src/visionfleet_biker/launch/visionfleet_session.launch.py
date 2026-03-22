from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

ASSETS = '/home/danny/VisionFleet_ws/assets'


def generate_launch_description():
    qty_arg = DeclareLaunchArgument(
        'quantity', default_value='2',
        description='Number of bikes to spawn'
    )
    quantity = LaunchConfiguration('quantity')

    spawn_bikes = ExecuteProcess(
        cmd=[
            'python3',
            f'{ASSETS}/fleet_spawner.py',
            '-f', f'{ASSETS}/bike_stack.json',
            '-q', quantity
        ],
        output='screen'
    )

    detection = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='visionfleet_biker',
                executable='detection_node.py',
                name='detection_node',
                output='screen'
            )
        ]
    )

    return LaunchDescription([qty_arg, spawn_bikes, detection])