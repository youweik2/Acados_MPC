#!/usr/bin/env python3

# Basic
import numpy as np
from scipy import signal
from math import *
import rospy
from filterpy.kalman import KalmanFilter

# Message
from std_msgs.msg import String, Bool, Float32, Float64
from geometry_msgs.msg import Pose2D

# GEM Sensor Headers
from septentrio_gnss_driver.msg import INSNavGeod
from sensor_msgs.msg import NavSatFix

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt

# MPC import
import pickle
import matplotlib.pyplot as plt
from ros1_MPC.BaseMPC import GemCarOptimizer
from ros1_MPC.GemCar import GemCarModel

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


class VehicleController():
    def __init__(self):

        self.rate       = rospy.Rate(20)
        self.look_ahead = 4
        self.wheelbase  = 2.57 # meters
        self.offset     = 1.26 # meters

        self.gnss_sub   = rospy.Subscriber("/septentrio_gnss/navsatfix", NavSatFix, self.gnss_callback)
        self.ins_sub    = rospy.Subscriber("/septentrio_gnss/insnavgeod", INSNavGeod, self.ins_callback)
        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)
        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)

        self.lat        = 0.0
        self.lon        = 0.0
        self.heading    = 0.0
        self.speed      = 0.0

        self.olat       = 40.092855    
        self.olon       = -88.235981 

        # -------------------- PACMod setup --------------------

        self.gem_enable    = False
        self.pacmod_enable = False

        # GEM vehicle enable, publish once
        self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
        self.enable_cmd = Bool()
        self.enable_cmd.data = False

        # GEM vehicle gear control, neutral, forward and reverse, publish once
        self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 2 # SHIFT_NEUTRAL

        # GEM vehilce brake control
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.brake_cmd = PacmodCmd()
        self.brake_cmd.enable = False
        self.brake_cmd.clear  = True
        self.brake_cmd.ignore = True

        # GEM vechile forward motion control
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.accel_cmd = PacmodCmd()
        self.accel_cmd.enable = False
        self.accel_cmd.clear  = True
        self.accel_cmd.ignore = True

        # GEM vechile turn signal control
        self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        self.turn_cmd = PacmodCmd()
        self.turn_cmd.ui16_cmd = 1 # None

        # GEM vechile steering wheel control
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)
        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0 # radians, -: clockwise, +: counter-clockwise
        self.steer_cmd.angular_velocity_limit = 3.5 # radians/second

        # MPC setting
        self.Epi = 3000
        self.plot_figures = True
        self.target_x = 0.0
        self.target_y = 50.0
        self.horizon = 1.0
        self.dt = 0.02
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

        # publish point info
        self.cord_pub = rospy.Publisher('cord', Pose2D,  queue_size=1)  
        self.kal_cord_pub = rospy.Publisher('kal_cord', Pose2D, queue_size=1)

        # Desired control values
        self.steering_angle = 0.0  # Steering wheel angle in radians
        self.steering_speed_limit = 3.5  # Steering wheel rotation speed in radians/sec
        self.brake_percent = 0.0    # Brake command (0.0 to 1.0)
        self.throttle_percent = 0.0 # Throttle command (0.0 to 1.0)

        # initial params
        self.lat = 0.0
        self.lon = 0.0
        self.heading = 0.0
        self.angle = 0.0
        self.speed = 0.0
        self.steer_angle = 0.0
        self.steer_delta = 0.0
        self.filtered_x, self.filtered_y, self.filtered_yaw = 0.0, 0.0, 0.0

        # log info
        self.x_log, self.y_log, self.theta_log = [], [], []
        self.x_real_log, self.y_real_log = [], []
        self.o_log, self.a_log = [], []

    def ins_callback(self, msg):
        self.heading = round(msg.heading, 6)
        self.angle = self.heading

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
    
    def wps_to_xy(self, lon_wp, lat_wp):
        # convert GNSS waypoints into local fixed frame reprented in x and y
        lon_wp_x, lat_wp_y = ll2xy(lat_wp, lon_wp, self.olat, self.olon)    # need to add source code to provide support
        return lon_wp_x, lat_wp_y

    def get_gem_state(self):

        # vehicle gnss heading (yaw) in degrees
        # vehicle x, y position in fixed local frame, in meters
        # reference point is located at the center of GNSS antennas
        local_x_curr, local_y_curr = self.wps_to_xy(self.lon, self.lat)

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        self.curr_yaw = self.heading_to_yaw(self.heading) 

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
    

    def start_loop(self):
        
        while not rospy.is_shutdown():

            if (self.gem_enable == False):
                if(self.pacmod_enable == True):

                    # ---------- enable PACMod ----------

                    # enable forward gear
                    self.gear_cmd.ui16_cmd = 3

                    # enable brake
                    self.brake_cmd.enable  = True
                    self.brake_cmd.clear   = False
                    self.brake_cmd.ignore  = False
                    self.brake_cmd.f64_cmd = 0.0

                    # enable gas 
                    self.accel_cmd.enable  = True
                    self.accel_cmd.clear   = False
                    self.accel_cmd.ignore  = False
                    self.accel_cmd.f64_cmd = 0.0

                    self.gear_pub.publish(self.gear_cmd)
                    print("Foward Engaged!")

                    self.turn_pub.publish(self.turn_cmd)
                    print("Turn Signal Ready!")
                    
                    self.brake_pub.publish(self.brake_cmd)
                    print("Brake Engaged!")

                    self.accel_pub.publish(self.accel_cmd)
                    print("Gas Engaged!")

                    self.gem_enable = True

            # Cord command
            cord = Pose2D()
            self.get_gem_state()
            cord.x = self.curr_x
            cord.y = self.curr_y
            cord.theta = self.curr_yaw
            self.cord_pub.publish(cord)

            kal_cord = Pose2D()
            self.kalman_filter()
            kal_cord.x = self.filtered_x
            kal_cord.y = self.filtered_y
            kal_cord.theta = self.filtered_yaw
            self.kal_cord_pub.publish(kal_cord)
            
            self.mpc_interface()
            self.rate.sleep()

    def publish_commands(self):

        if (self.steer_delta <= 30 and self.steer_delta >= -30):
            self.turn_cmd.ui16_cmd = 1
        elif(self.steer_delta > 30):
            self.turn_cmd.ui16_cmd = 2 # turn left
        else:
            self.turn_cmd.ui16_cmd = 0 # turn right
        
        self.accel_cmd.f64_cmd = self.accel_percent
        self.steer_cmd.angular_position = np.radians(self.steer_angle)
        self.accel_pub.publish(self.accel_cmd)
        self.steer_pub.publish(self.steer_cmd)
        self.turn_pub.publish(self.turn_cmd)

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

        speed_controller = PID(0.5, 0.01, 0.05, 5)
        
        # Signal Filter
        vel_filted = []
        b, a = signal.butter(2, 1.0, btype="low", fs=1000.0)
        zi = signal.lfilter_zi(b, a) * self.speed
        
        # initial State
        x_real, y_real, vel_real = self.filtered_x, self.filtered_y, self.speed
        theta_real = self.angle
             
        x_0, y_0, theta_0 = x_real, y_real, theta_real  # Save the initial theta

        # Solver
        x_0, y_0, theta_0, vel, a_0, o_0 = opt.solve(x_real, y_real, theta_real, vel_real)

        x_real, y_real, theta_real, vel_real = x_0, y_0, theta_0, vel
        target_a, target_o = a_0, o_0

        self.x_log.append(x_0)
        self.y_log.append(y_0)
        self.theta_log.append(theta_0)
        self.x_real_log.append(x_real)
        self.y_real_log.append(y_real)
        self.a_log.append(a_0)
        self.o_log.append(o_0)
        
        # Signal Filter
        z, zi = signal.lfilter(b, a, [self.speed], zi=zi)
        vel_filted.append(z) # use for plotting

        current_time = rospy.get_time()

        # PID Control
        expected_acceleration = speed_controller.get_control(current_time, target_a - z)

        # Publish control commands
        self.accel_percent = expected_acceleration
        # self.throttle_percent, self.brake_percent = self.accel2ctrl(expected_acceleration)
        
        self.delta = np.degrees(round(np.clip(target_o, -0.61, 0.61), 3))
        self.steer_angle = self.front2steer(self.steer_delta)

        # self.publish_commands(brake_percent, throttle_percent, steer_angle)
        self.publish_commands()

        # Terminal State: Stop iff reached
        if (x_0 - self.target_x) ** 2 + (y_0 - self.target_y) ** 2 < 0.25:
            # break
            rospy.loginfo("Stopping the node...")
            rospy.signal_shutdown("Reach the target")

        # boundary condition
        if x_0 < -10 or x_0 > 10 or y_0 > 50 or y_0 < -50:
            # break
            rospy.loginfo("Stopping the node...")
            rospy.signal_shutdown("Exceed the bounds")

    # plot function
    def plot_results(self, start_x, start_y, theta_log, a_log, x_log, y_log, x_real_log, y_real_log, o_log):
        
        plt.figure()
        a = np.arange(0, (len(a_log)), 1)*self.dt
        plt.plot(a, a_log, 'r-', label='desired a')
        plt.xlabel('time')
        plt.ylabel('value')
        plt.legend()
        plt.grid(True)
        plt.show()

        v = np.arange(0, (len(o_log)), 1)*self.dt
        plt.plot(v, o_log, 'r-', label='desired theta')
        plt.xlabel('time')
        plt.ylabel('value')
        plt.legend()
        plt.grid(True)
        plt.show()        

        # Plot for angles
        t = np.arange(0, (len(theta_log)), 1)*self.dt
        plt.plot(t, theta_log, 'r-', label='desired theta')
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

def main():
    rospy.init_node('mpc_node', anonymous=True)
    rospy.loginfo("MPC Node Start")
    controller = VehicleController()

    try:
        controller.start_loop()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()