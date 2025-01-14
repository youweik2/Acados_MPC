import rclpy
from rclpy.node import Node
import numpy as np
from scipy import signal
from math import *
from rclpy.node import Node
from rclpy.clock import Clock
from std_msgs.msg import String, Bool, Float32, Float64
from pacmod2_msgs.msg import GlobalCmd, PositionWithSpeed, VehicleSpeedRpt, SystemCmdFloat, SystemCmdInt
from septentrio_gnss_driver.msg import INSNavGeod
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import NavSatFix
## MPC
from 25mpc.BaseMPC import GemCarOptimizer
from 25mpc.GemCar import GemCarModel

## LL2XY ##
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


class PID:
    def __init__(self, Kp=0.0, Ki=0.0, Kd=0.0, windup_thres=0.0):
        self.current_error = None
        self.past_error = None
        self.integral_error = 0.0
        self.derivative_error = None

        self.current_time = 0.0
        self.past_time = None

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.windup_thres = windup_thres

    def reset(self):
        self.current_error = None
        self.past_error = None
        self.integral_error = 0.0
        self.derivative_error = None

        self.current_time = 0.0
        self.past_time = None

        self.Kp = 0.0
        self.Ki = 0.0
        self.Kd = 0.0

        self.windup_thres = 0.0

    def get_control(self, current_time, current_error):
        self.current_time = current_time
        self.current_error = current_error

        if self.past_time is None:
            self.past_time = self.current_time
            expected_acceleration = 0.0
        else:
            self.integral_error += self.current_error * (
                self.current_time - self.past_time
            )
            self.derivative_error = (self.current_error - self.past_error) / (
                self.current_time - self.past_time
            )
            np.clip(self.integral_error, -self.windup_thres, self.windup_thres)
            expected_acceleration = (
                self.Kp * self.current_error
                + self.Ki * self.integral_error
                + self.Kd * self.derivative_error
            )
        self.past_time = self.current_time
        self.past_error = self.current_error

        return expected_acceleration


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
        self.angle = 0.0
        self.speed = 0.0

        self.olat = 40.092855
        self.olon = -88.235981

        self.command_timer = self.create_timer(0.1, self.publish_commands)  # 0.1 means it will be called every 0.1 seconds
        self.clock = Clock()
        self.current_time = self.clock.now()

        # MPC setting
        self.Epi = 3000
        self.plot_figures = False
        self.target_x = 0.0
        self.target_y = 50.0
        self.horizon = 1.0
        self.dt = 0.05

    def ins_callback(self, msg):
        self.heading = round(msg.heading, 6)
        self.angle = (self.heading- 90.0)*np.pi/180

    def gnss_callback(self, msg):
        self.lat = round(msg.latitude, 6)
        self.lon = round(msg.longitude, 6)

    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def heading2yaw(self, heading_curr):
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
    
    def wps2xy(self, lon_wp, lat_wp):
        # convert GNSS waypoints into local fixed frame reprented in x and y
        lon_wp_x, lat_wp_y = ll2xy(lat_wp, lon_wp, self.olat, self.olon)    # need to add source code to provide support
        return lon_wp_x, lat_wp_y
    
        
    def accel2ctrl(self, expected_acceleration):
        if expected_acceleration >= -0.2:
            throttle_percent = 0.1187 * expected_acceleration + 0.2600
            brake_percent = 0.0
        else:
            throttle_percent = 0.0
            brake_percent = -0.1945 * expected_acceleration + 0.2421

        throttle_percent = np.clip(throttle_percent, 0.0, 0.45)
        brake_percent = np.clip(brake_percent, 0.0, 0.5)

        return throttle_percent, brake_percent

    def get_gem_state(self):
        # vehicle gnss heading (yaw) in degrees
        # vehicle x, y position in fixed local frame, in meters
        # reference point is located at the center of GNSS antennas
        local_x_curr, local_y_curr = self.wps2xy(self.lon, self.lat)

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        curr_yaw = self.heading2yaw(self.heading) 

        # reference point is located at the center of rear axle
        curr_x = local_x_curr - self.offset * np.cos(curr_yaw)
        curr_y = local_y_curr - self.offset * np.sin(curr_yaw)

        return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)

    def publish_commands(self, brake_value, accel_value, steer_value):

        # Cord command
        cord = Pose2D()
        curr_x, curr_y, curr_yaw = self.get_gem_state()
        cord.x = curr_x
        cord.y = curr_y
        cord.theta = curr_yaw
        self.cord_pub.publish(cord)

        # publish
        self.accel_cmd.command = accel_value
        self.accel_pub.publish(self.accel_cmd)

        self.brake_cmd.command = brake_value
        self.brake_pub.publish(self.accel_cmd)       

        self.steer_cmd.angular_position = steer_value
        self.steer_cmd.angular_velocity_limit = self.steering_speed_limit
        self.steer_pub.publish(self.steer_cmd)

        self.global_pub.publish(self.global_cmd)


    def mpc_interface(self):
        
        self.obstacles = np.array([
            [0.0, 20, 1],        #x, y, r 20 25 30
            [-1.0, 25, 1],
            [1.0, 30, 1]
            ])

        # Car model and PID
        car_model = GemCarModel()
        opt = GemCarOptimizer(m_model=car_model.model, 
                                m_constraint=car_model.constraint, t_horizon=self.horizon, dt=self.dt, obstacles = self.obstacles)

        speed_controller = PID(5.0, 0.0, 2, 10)
        speed_controller_second = PID(4.2, 0.5, 0.2, 30)
        
         # Solver
        x_0, y_0, theta, X, U = opt.solve(x_real, y_real, theta_real)
        target_v, target_o = U.T[0], U.T[1]
                                    
        # Signal Filter
        z, zi = signal.lfilter(b, a, [self.speed], zi=zi)

                                    # PID Control
        if z < target_v - 0.2 or z > target_v + 0.2:
            expected_acceleration = speed_controller.get_control(self.current_time, target_v - z)
            speed_controller_second.integral_error = 0.0
        else:
            expected_acceleration = speed_controller_second.get_control(self.current_time, target_v - z)

        # Publish control commands

        throttle_percent, brake_percent = self.accel2ctrl(expected_acceleration)
        isteer = (self.wheelbase * target_o) / (self.speed * np.pi) if abs(self.speed) > 0.05 else 0.0
        steer_angle = self.front2steer(isteer)

        self.publish_commands(brake_percent, throttle_percent, steer_angle)

        # Next Step

        x_real, y_real, curr_yaw = self.get_gem_state()
        theta_real = self.angle



def main(args=None):
    rclpy.init(args=args)
    vehicle_controller = VehicleController()
    rclpy.spin(vehicle_controller)
    vehicle_controller.mpc_interface()
    vehicle_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
