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

from filterpy.kalman import KalmanFilter

# MPC import
import pickle
import matplotlib.pyplot as plt
from ctrl.ros2_MPC import GemCarOptimizer, GemCarModel
from tqdm import tqdm



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


class VehicleController_MPC(Node):
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

        # Desired control values
        self.steering_angle = 0.0  # Steering wheel angle in radians
        self.steering_speed_limit = 3.5  # Steering wheel rotation speed in radians/sec
        self.brake_percent = 0.0    # Brake command (0.0 to 1.0)
        self.throttle_percent = 0.0 # Throttle command (0.0 to 1.0)

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

        # MPC setting
        self.Epi = 3000
        self.plot_figures = True
        self.target_x = 0.0
        self.target_y = 50.0
        self.horizon = 1.0
        self.dt = 0.05
        self.circle_obstacles_1 = {'x': 0, 'y': 20, 'r': 1.0}
        self.circle_obstacles_2 = {'x': 1, 'y': 25, 'r': 1.0}
        self.circle_obstacles_3 = {'x': -1, 'y': 30, 'r': 1.0}

        # Kalman filter initialization for x, y, yaw
        self.kf = KalmanFilter(dim_x=3, dim_z=3)
        self.kf.x = np.array(self.get_gem_state())  # Initial state: [x, y, yaw]
        self.kf.F = np.eye(3)  # State transition matrix
        self.kf.H = np.eye(3)  # Measurement function
        self.kf.P *= 1000.0  # Covariance matrix
        self.kf.R = np.eye(3) * 5  # Measurement noise
        self.kf.Q = np.eye(3) * 0.1  # Process noise

        self.clock = Clock()
        self.current_time = self.clock.now()

        self.command_timer = self.create_timer(0.1, self.publish_commands)  # 0.1 means it will be called every 0.1 seconds
        


    def ins_callback(self, msg):
        self.heading = round(msg.heading, 6)
        self.angle = self.heading #- np.pi/2

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
            self.brake_cmd.command = self.brake_percent
            self.accel_cmd.command = self.throttle_percent

            self.steer_cmd.angular_position = 0.0 # radians, -: clockwise, +: counter-clockwise
            self.steer_cmd.angular_velocity_limit = 3.5 # radians/second

            self.global_pub.publish(self.global_cmd)
            self.gear_pub.publish(self.gear_cmd)
            self.brake_pub.publish(self.brake_cmd)
            self.accel_pub.publish(self.accel_cmd)

            self.gem_enable = True
        
        # needed to enable pacmod, do not remove!
        # self.global_cmd.enable = True
        # self.global_cmd.clear_override  = False
        # self.global_cmd.ignore_override = False

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

        self.mpc_interface()

        # publish
        self.accel_cmd.command = self.throttle_percent
        self.accel_pub.publish(self.accel_cmd)

        self.brake_cmd.command = self.brake_percent
        self.brake_pub.publish(self.accel_cmd)       

        self.steer_cmd.angular_position = self.steer_value
        self.steer_cmd.angular_velocity_limit = self.steering_speed_limit
        self.steer_pub.publish(self.steer_cmd)

        self.global_pub.publish(self.global_cmd)

        # self.get_logger().info(
        #     f'Steering Angle: {self.steering_angle:.2f} rad, '
        #     f'Acceleration: {self.throttle_percent:.2f}, '
        #     f'Brake: {self.brake_percent:.2f}'
        # )

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
        
        # Signal Filter
        vel_filted = []
        b, a = signal.butter(2, 1.0, btype="low", fs=1000.0)
        zi = signal.lfilter_zi(b, a) * self.speed
        
        # initial State
        x_real, y_real = self.filtered_x, self.filtered_y
        theta_real = self.angle
        # x_real, y_real= 0, 0
        # theta_real = np.pi/2

        start_x, start_y = x_real, y_real                
        x_0, y_0, theta = x_real, y_real, theta_real
        theta_0 = theta_real        # Save the initial theta
        U_real = np.array([0.0, 0.0]) # U_real

        try:
            # Solver
            x_0, y_0, theta, X, U = opt.solve(x_real, y_real, theta_real)

            # target_v, target_o = U.T[0], U.T[1]
            target_v, target_o = U[0][0], U[0][1]
            
            # Signal Filter
            z, zi = signal.lfilter(b, a, [self.speed], zi=zi)
            vel_filted.append(z) # use for plotting

            # PID Control
            if z < target_v - 0.2 or z > target_v + 0.2:
                expected_acceleration = speed_controller.get_control(
                    self.current_time.nanoseconds, target_v - z
                )
                speed_controller_second.integral_error = 0.0
            else:
                expected_acceleration = speed_controller_second.get_control(
                    self.current_time.nanoseconds, target_v - z
                )

            # Publish control commands

            self.throttle_percent, self.brake_percent = self.accel2ctrl(expected_acceleration)
            isteer = (self.wheelbase * target_o) / (self.speed * np.pi) if abs(self.speed) > 0.05 else 0.0
            self.steer_angle = self.front2steer(isteer)

            # self.publish_commands(brake_percent, throttle_percent, steer_angle)
            self.publish_commands()

            # Next Step
            x_real, y_real = self.filtered_x, self.filtered_y
            theta_real = self.angle
            # x_real, y_real, theta_real = x_0, y_0, theta

            # Terminal State: Stop iff reached

            if (x_0 - self.target_x) ** 2 + (y_0 - self.target_y) ** 2 < 0.1:
                # break
                print("reach the target", theta_0)
                return

        except RuntimeError:
            print("Infesible", theta_0)
            return

        print("not reach the target", theta_0)

    def plot_results(self, start_x, start_y, theta_log, U_log, x_log, y_log, x_real_log, y_real_log, U_real_log, theta_real_log):
        
        plt.figure()
        tt = np.arange(0, (len(U_log)), 1)*self.dt
        t = np.arange(0, (len(theta_log)), 1)*self.dt
        plt.plot(tt, U_log, 'r-', label='desired U')
        plt.plot(tt, U_real_log, 'b-', label='U_real', linestyle='--')
        plt.xlabel('time')
        plt.ylabel('U')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot for angles
        plt.plot(t, theta_log, 'r-', label='desired theta')

        # plt.plot(t, theta_real_log, 'b-', label='theta_real')
        plt.xlabel('time')
        plt.ylabel('theta')
        plt.legend()
        plt.grid(True)
        plt.show()

        ## Plot for circle obstacles and x-y positions
        
        plt.plot(x_log, y_log, 'r-', label='desired path')
        plt.plot(x_real_log, y_real_log, 'b-', label='real path', linestyle='--')
        plt.plot(self.target_x,self.target_y,'bo')
        plt.plot(start_x, start_y, 'go')
        plt.xlabel('pos_x')
        plt.ylabel('pos_y')

        target_circle1 = plt.Circle((self.circle_obstacles_1['x'], self.circle_obstacles_1['y']), self.circle_obstacles_1['r'], color='whitesmoke', fill=True)
        target_circle2 = plt.Circle((self.circle_obstacles_2['x'], self.circle_obstacles_2['y']), self.circle_obstacles_2['r'], color='whitesmoke', fill=True)
        target_circle3 = plt.Circle((self.circle_obstacles_3['x'], self.circle_obstacles_3['y']), self.circle_obstacles_3['r'], color='whitesmoke', fill=True)
        target_circle4 = plt.Circle((self.circle_obstacles_1['x'], self.circle_obstacles_1['y']), self.circle_obstacles_1['r'], color='k', fill=False)
        target_circle5 = plt.Circle((self.circle_obstacles_2['x'], self.circle_obstacles_2['y']), self.circle_obstacles_2['r'], color='k', fill=False)
        target_circle6 = plt.Circle((self.circle_obstacles_3['x'], self.circle_obstacles_3['y']), self.circle_obstacles_3['r'], color='k', fill=False)
        
        plt.gcf().gca().add_artist(target_circle1)
        plt.gcf().gca().add_artist(target_circle2)
        plt.gcf().gca().add_artist(target_circle3)
        plt.gcf().gca().add_artist(target_circle4)
        plt.gcf().gca().add_artist(target_circle5)
        plt.gcf().gca().add_artist(target_circle6)
        plt.axis('equal')
        plt.legend()
        plt.show()

        with open('single_traj_mpc_50hz.pkl', 'wb') as f:
            pickle.dump([x_log, y_log], f)



def main(args=None):
    rclpy.init(args=args)
    vehicle_controller = VehicleController_MPC()
    rclpy.spin(vehicle_controller)
    # vehicle_controller.mpc_interface()
    vehicle_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
