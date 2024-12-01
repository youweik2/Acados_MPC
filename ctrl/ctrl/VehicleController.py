import rclpy
from rclpy.node import Node
import numpy as np
from scipy import signal
from math import *
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32, Float64
from pacmod2_msgs.msg import GlobalCmd, PositionWithSpeed, VehicleSpeedRpt, SystemCmdFloat, SystemCmdInt
from septentrio_gnss_driver.msg import INSNavGeod
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import NavSatFix
from filterpy.kalman import KalmanFilter


def mdeglat(lat):
    latrad = lat*2.0*pi/360.0
    dy = 111132.09 - 566.05 * cos(2.0*latrad) \
         + 1.20 * cos(4.0*latrad) \
         - 0.002 * cos(6.0*latrad)
    return dy

def mdeglon(lat):
    latrad = lat*2.0*pi/360.0 
    dx = 111415.13 * cos(latrad) \
         - 94.55 * cos(3.0*latrad) \
	+ 0.12 * cos(5.0*latrad)
    return dx

def ll2xy(lat, lon, orglat, orglon):
    x = (lon - orglon) * mdeglon(orglat)
    y = (lat - orglat) * mdeglat(orglat)
    return (x,y)


class VehicleController(Node):
    def __init__(self):
        super().__init__('vehicle_controller')
         # Subscribers
        self.gnss_sub = self.create_subscription(NavSatFix, '/navsatfix', self.gnss_callback, 10)
        self.ins_sub = self.create_subscription(INSNavGeod, '/insnavgeod', self.ins_callback, 10)
        self.enable_sub = self.create_subscription(Bool, '/pacmod/enabled', self.enable_callback, 10)
        self.speed_sub = self.create_subscription(VehicleSpeedRpt, '/pacmod/vehicle_speed_rpt', self.speed_callback, 10)

        # Publishers for steering, acceleration, braking, and gear commands
        self.global_pub = self.create_publisher(GlobalCmd, '/pacmod/global_cmd', 1)
        self.steer_pub = self.create_publisher(PositionWithSpeed, '/pacmod/steering_cmd', 1)
        self.accel_pub = self.create_publisher(SystemCmdFloat, '/pacmod/accel_cmd', 1)
        self.brake_pub = self.create_publisher(SystemCmdFloat, '/pacmod/brake_cmd', 1)
        self.gear_pub = self.create_publisher(SystemCmdInt, '/pacmod/shift_cmd', 1)

        # publish point info
        self.cord_pub = self.create_publisher(Pose2D, 'cord', 10)
        self.kal_cord_pub = self.create_publisher(Pose2D, 'kal_cord', 10)

        self.declare_parameter('acceleration', 0.0)
        self.declare_parameter('steering_angle', 0.0)

        # Desired control values
        self.steering_angle = self.get_parameter('steering_angle').value  # Steering wheel angle in radians
        self.steering_speed_limit = 3.5  # Steering wheel rotation speed in radians/sec
        self.acceleration = self.get_parameter('acceleration').value    # Throttle command (0.0 to 1.0)
        self.brake = 0.0           # Brake command (0.0 to 1.0)

        # Initialize PACMod command messages
        self.global_cmd = GlobalCmd()
        self.global_cmd.enable = False
        self.global_cmd.clear_override  = True
        self.global_cmd.ignore_override = True

        self.gear_cmd = SystemCmdInt()
        self.gear_cmd.command = 2 # SHIFT_NEUTRAL
        self.brake_cmd = SystemCmdFloat()
        self.accel_cmd = SystemCmdFloat()

        self.gem_enable = False
        self.pacmod_enable = True

        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0 # radians, -: clockwise, +: counter-clockwise
        self.steer_cmd.angular_velocity_limit = 3.5 # radians/second

        self.wheelbase  = 2.57 # meters
        self.offset     = 1.26 # meters

        self.lat = 0.0
        self.lon = 0.0
        self.heading = 0.0

        self.speed = 0.0

        self.olat = 40.09281153008717
        self.olon = -88.23607685860453

        # Kalman filter initialization for x, y, yaw
        self.kf = KalmanFilter(dim_x=3, dim_z=3)
        self.kf.x = np.array([0.0, 0.0, 0.0])  # Initial state: [x, y, yaw]
        self.kf.F = np.eye(3)  # State transition matrix
        self.kf.H = np.eye(3)  # Measurement function
        self.kf.P *= 1000.0  # Covariance matrix
        self.kf.R = np.eye(3) * 5  # Measurement noise
        self.kf.Q = np.eye(3) * 0.1  # Process noise

        self.command_timer = self.create_timer(0.1, self.publish_commands)  # 0.1 means it will be called every 0.1 seconds
    
    def ins_callback(self, msg):
        self.heading = round(msg.heading, 6)

    def gnss_callback(self, msg):
        self.lat = round(msg.latitude, 6)
        self.lon = round(msg.longitude, 6)

    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def heading_to_yaw(self, heading_curr):
        if (heading_curr >= 270 and heading_curr < 360):
            yaw_curr = np.radians(450 - heading_curr)
        else:
            yaw_curr = np.radians(90 - heading_curr)
        return yaw_curr
    
    def front2steer(self, f_angle):
        if(f_angle > 35):
            f_angle = 35
        if (f_angle < -35):
            f_angle = -35
        if (f_angle > 0):
            steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        elif (f_angle < 0):
            f_angle = -f_angle
            steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        else:
            steer_angle = 0.0
        return steer_angle
    
    def wps_to_local_xy(self, lon_wp, lat_wp):
        # convert GNSS waypoints into local fixed frame reprented in x and y
        lon_wp_x, lat_wp_y = ll2xy(lat_wp, lon_wp, self.olat, self.olon)    # need to add source code to provide support
        return lon_wp_x, lat_wp_y

    def get_gem_state(self):
        # vehicle gnss heading (yaw) in degrees
        # vehicle x, y position in fixed local frame, in meters
        # reference point is located at the center of GNSS antennas
        local_x_curr, local_y_curr = self.wps2xy(self.lon, self.lat)

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        self.curr_yaw = self.heading2yaw(self.heading) 

        # reference point is located at the center of rear axle
        self.curr_x = local_x_curr - self.offset * np.cos(self.curr_yaw)
        self.curr_y = local_y_curr - self.offset * np.sin(self.curr_yaw)

    def kalman_filter(self):
        # Kalman filter update
        z = np.array([self.curr_x, self.curr_y, self.curr_yaw])  # Measurement
        self.kf.predict()
        self.kf.update(z)

        # Get the filtered state
        self.filtered_x, self.filtered_y, self.filtered_yaw = self.kf.x

    def publish_commands(self):
        if not self.gem_enable and self.pacmod_enable:
            self.get_logger().info('enable pacmod')
            # ---------- enable PACMod ----------
            self.global_cmd.enable = True
            self.global_cmd.clear_override  = False
            self.global_cmd.ignore_override = False

            self.gear_cmd.command = 3  # enable forward gear
            self.brake_cmd.command = 0.0
            self.accel_cmd.command = 0.0

            self.global_pub.publish(self.global_cmd)
            self.gear_pub.publish(self.gear_cmd)
            self.brake_pub.publish(self.brake_cmd)
            self.accel_pub.publish(self.accel_cmd)

            self.gem_enable = True
        
        self.global_cmd.enable = True
        self.global_cmd.clear_override  = False
        self.global_cmd.ignore_override = False

        # Cord command
        cord = Pose2D()
        self.get_gem_state()
        cord.x = self.curr_x
        cord.y = self.curr_y
        cord.theta = self.curr_yaw
        self.cord_pub.publish(cord)

        kal_cord = Pose2D()
        self.kalman_filter()
        kal_cord.x = self.filter_x
        kal_cord.y = self.filter_y
        kal_cord.theta = self.filter_yaw
        self.kal_cord_pub.publish(kal_cord)

        # for testing
        self.accel_cmd.command = self.acceleration
        self.accel_pub.publish(self.accel_cmd)

        self.steer_cmd.angular_position = self.steering_angle
        self.steer_cmd.angular_velocity_limit = self.steering_speed_limit
        self.steer_pub.publish(self.steer_cmd)

        self.global_pub.publish(self.global_cmd)


        # self.get_logger().info(
        #     f'Steering Angle: {self.steering_angle:.2f} rad, '
        #     f'Acceleration: {self.acceleration:.2f}, '
        #     f'Brake: {self.brake:.2f}, '
        #     f'Gear: {self.gear}'
        # )

def main(args=None):
    rclpy.init(args=args)
    vehicle_controller = VehicleController()
    rclpy.spin(vehicle_controller)
    vehicle_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
