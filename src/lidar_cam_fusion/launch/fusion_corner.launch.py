import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('lidar_cam_fusion')

    default_config = os.path.join(pkg_dir, 'config', 'fusion_params.yaml')
    default_rviz = os.path.join(pkg_dir, 'rviz', 'fusion.rviz')

    # Launch arguments
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_config,
        description='Path ke file parameter YAML'
    )

    launch_camera_arg = DeclareLaunchArgument(
        'launch_camera',
        default_value='true',
        description='Jalankan camera driver node otomatis (/dev/video*) (true/false)'
    )

    video_device_arg = DeclareLaunchArgument(
        'video_device',
        default_value='/dev/video2',
        description='Device path atau index kamera (mis. /dev/video2 atau 2)'
    )

    detector_arg = DeclareLaunchArgument(
        'detector',
        default_value='corner',
        description='Jenis detektor: corner (Klasik 3D 35x35x35 cm tanpa AI), yolo, atau hsv'
    )

    show_preview_arg = DeclareLaunchArgument(
        'show_preview',
        default_value='false',
        description='Buka jendela preview langsung GUI OpenCV terpisah (true/false, default: false agar fokus di RViz)'
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Buka RViz2 dengan visualisasi kamera, titik 3D, dan kubus 35cm (true/false)'
    )

    lidar_pitch_deg_arg = DeclareLaunchArgument(
        'lidar_pitch_deg',
        default_value='-33.0',
        description='Sudut kemiringan LiDAR ke bawah dalam derajat (CAD bracket: -33.0 deg)'
    )

    publish_static_tf_arg = DeclareLaunchArgument(
        'publish_static_tf',
        default_value='true',
        description='Mempublikasikan TF statis odom -> base_link -> livox_frame & camera_optical_frame untuk RViz'
    )

    # 1. Node Pengambil Gambar Kamera (Logitech MX Brio)
    camera_node = Node(
        package='lidar_cam_fusion',
        executable='camera_node',
        name='camera_publisher_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'video_device': LaunchConfiguration('video_device')
            }
        ],
        condition=IfCondition(LaunchConfiguration('launch_camera'))
    )

    # 2. Node Fusi Kamera + LiDAR (Mode Detektor Corner 3D 35x35x35 cm)
    fusion_node = Node(
        package='lidar_cam_fusion',
        executable='fusion_node',
        name='lidar_cam_fusion_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'detector': LaunchConfiguration('detector'),
                'show_preview': LaunchConfiguration('show_preview'),
                'lidar_pitch_deg': LaunchConfiguration('lidar_pitch_deg')
            }
        ]
    )

    # 3. Static Transform Publishers (Agar TF di RViz valid & tidak merah No TF Data)
    # odom -> base_link
    tf_odom_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_odom_base',
        arguments=['--frame-id', 'odom', '--child-frame-id', 'base_link'],
        condition=IfCondition(LaunchConfiguration('publish_static_tf'))
    )

    # base_link -> livox_frame (LiDAR berada 8.2 cm di atas kamera, miring 33 deg ke depan robot)
    tf_base_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_base_lidar',
        arguments=['--x', '0.039', '--y', '0.0', '--z', '0.132',
                   '--qx', '0.200829', '--qy', '0.200829', '--qz', '0.677988', '--qw', '0.677988',
                   '--frame-id', 'base_link', '--child-frame-id', 'livox_frame'],
        condition=IfCondition(LaunchConfiguration('publish_static_tf'))
    )

    # base_link -> camera_optical_frame (Kamera berada 5 cm dari base_link menghadap lurus ke depan robot)
    # Konvensi ROS REP 103: Z_cam maju (+X_base), X_cam kanan (-Y_base), Y_cam bawah (-Z_base)
    tf_base_cam = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_base_cam',
        arguments=['--x', '0.05', '--y', '0.0', '--z', '0.05',
                   '--qx', '-0.5', '--qy', '0.5', '--qz', '-0.5', '--qw', '0.5',
                   '--frame-id', 'base_link', '--child-frame-id', 'camera_optical_frame'],
        condition=IfCondition(LaunchConfiguration('publish_static_tf'))
    )

    # 4. Node RViz2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_fusion',
        arguments=['-d', default_rviz],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    return LaunchDescription([
        params_file_arg,
        launch_camera_arg,
        video_device_arg,
        detector_arg,
        show_preview_arg,
        rviz_arg,
        lidar_pitch_deg_arg,
        publish_static_tf_arg,
        camera_node,
        fusion_node,
        tf_odom_base,
        tf_base_lidar,
        tf_base_cam,
        rviz_node
    ])
