#!/usr/bin/env python

'''
Particle Filter bare bones implementation for 6.141 Lab 5 

Uses RangeLibc for calculating the sensor model:
https://github.com/kctess5/range_libc

'''


import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, Point32, PoseWithCovarianceStamped, PointStamped
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from std_msgs.msg import Header
import tf.transformations
import tf
import range_libc


### Useful functions
def angle_to_quaternion(angle):
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, angle))

def make_header(frame_id):
    stamp = rospy.Time.now()
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return header

def particle_to_pose(particle):
    pose = Pose()
    pose.position.x = particle[0]
    pose.position.y = particle[1]
    pose.orientation = angle_to_quaternion(particle[2])
    return pose

def particles_to_poses(particles):
    return map(particle_to_pose, particles)

def quaternion_to_angle(q):
    roll, pitch, yaw = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
    return yaw

def rotation_matrix_2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.matrix([[c, -s], [s, c]])


### Particle filter class
class ParticleFiler():
    def __init__(self):
        ###### Tunning Parameters ######
        # Motion model tunning parameters
        self.X_VARIANCE = .008
        self.Y_VARIANCE = .1
        self.X_MEAN_POS = 0.01
        self.Y_MEAN_POS = 0
        self.THETA_VARIANCE = 0.01

        # Sensor model tunning parameters
        self.sensor_model_slope = .00005
        self.sensor_model_spike = 0.03
        self.sensor_model_sigma = 10
        self.sensor_model_bias = 0.01

        # particle filter parameters
        self.MAX_PARTICLES = 6000
        self.MAX_RANGE_METERS = 10  # meters
        self.MAX_RANGE_PX = None
        # cddt is a discrete method. defines number of discrete thetas
        self.THETA_DISCRETIZATION = 60
        self.show_particles = False

        ###### Class storage containers ######
        # initialization checks
        self.lidar_initialized = False
        self.odom_initialized = False
        self.clicked_pose_initialized = False
        self.first_sensor_update = True

        # various data containers used in the MCL algorithm
        self.map_info = None
        bself.map_initialized = False

        # rangeLibc method
        self.range_method = None

        # variables to store particles and weights
        # weights are initialized and normalized to be the same values
        self.particles = np.zeros((self.MAX_PARTICLES, 3))
        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)

        # variables to store sensor data
        self.prevOdom = None
        self.odomAction = None
        self.queries = None
        self.laser_ranges = None
        self.sensor_model_table = None
        self.laser_angles = None

        ###### ROS Pub/Sub variables ######
        # publish particle visualization
        self.particle_pub  = rospy.Publisher("/particles", PoseArray, queue_size = 1)

        # input data: Laser and dead reckoning
        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.lidarCB, queue_size=1)
        self.odom_sub  = rospy.Subscriber("/odom", Odometry, self.odomCB, queue_size=1)

        # get initial pose estimate from rviz
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.clicked_pose, queue_size=1)
        self.click_sub = rospy.Subscriber("/clicked_point", PointStamped, self.clicked_pose, queue_size=1)

        # pose estimate output
        self.pose_estimate = None
        self.odom_pub = rospy.Publisher("/estimate_odom", Odometry, queue_size=2)

        ###### Run initialization functions ######
        self.get_omap()
        self.precompute_sensor_model()


        print "Initialization complete"

    def get_omap(self):
        """
             Loads the map and initializes RangeLibc
             Map server must be running
        """
        print("Loading Map")
        map_service_name = "static_map"
        rospy.wait_for_service(map_service_name)
        map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map

        self.map_info = map_msg.info
        print(self.map_info)
        oMap = range_libc.PyOMap(map_msg)
        # max range used internally in RangeLibc
        self.MAX_RANGE_PX = int(self.MAX_RANGE_METERS / self.map_info.resolution)

        # initialize range method
        self.range_method = range_libc.PyCDDTCast(oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
        print "Done loading map"
        self.map_initialized = True

    def publish_pose_estimate(self,pose):
        """
            Converts from pose estimate to odometry messsage
        """
        odom = Odometry()
        odom.header = make_header("map")
        odom.pose.pose.position.x = pose[0]
        odom.pose.pose.position.y = pose[1]
        odom.pose.pose.orientation = angle_to_quaternion(pose[2])
        self.odom_pub.publish(odom)

    def visualize(self):
        """
            Shows particles on Rviz
        """
        if self.particle_pub.get_num_connections() > 0 and self.show_particles:
            pa = PoseArray()
            pa.header = make_header("map")
            pa.poses = particles_to_poses(self.particles)
            self.particle_pub.publish(pa)


    def lidarCB(self, msg):
        """
            Callback for lidar data.
            Stores data on class containers
        """
        if not isinstance(self.laser_angles, np.ndarray):
            print "...Received first LiDAR message"
            self.laser_angles_full = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
            self.downsampled_angles = self.laser_angles_full[::10]
            self.laser_angles = self.laser_angles_full[::int(len(msg.ranges)/self.THETA_DISCRETIZATION)]
            self.lidar_initialized = True

        self.laser_ranges = np.array(msg.ranges)[::int(len(msg.ranges)/self.THETA_DISCRETIZATION)]

    def odomCB(self, msg):
        """
            Callback for Odometry data.
            Stores data on class containers and updates the particle filter
            odometry data is slower than lidar so we update the filter at the speed of odom data
        """
        if not self.odom_initialized:
            print "...Received first Odom message"
            self.prevOdom = msg
            self.odom_initialized = True
        else:
            # Get the rotation matrix of previous pose
            yaw_last = quaternion_to_angle(self.prevOdom.pose.pose.orientation)
            yaw_curr = quaternion_to_angle(msg.pose.pose.orientation)
            rotation_matrix = rotation_matrix_2d(yaw_last)
            # Now calculate the action (Curr Odom - Last Odom) and rotate
            d_x = msg.pose.pose.position.x - self.prevOdom.pose.pose.position.x
            d_y = msg.pose.pose.position.y - self.prevOdom.pose.pose.position.y
            d_yaw = yaw_curr - yaw_last
            d_pos = np.dot(rotation_matrix,np.array([d_x,d_y]).T)
            self.odomAction = np.array([d_pos[0,0], d_pos[0,1], d_yaw])
            self.prevOdom = msg
            self.update_particle_filter()

    def clicked_pose(self, msg):
        """
            Initializes the particles when a pose is clicked on rviz
            if the input is not a pose, we fill the entire map with particles (solving kidnapped robot problem)
        """
        if isinstance(msg, PoseWithCovarianceStamped):
            print "initialized pose 2d"
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            theta = quaternion_to_angle(msg.pose.pose.orientation)
            self.particles[:] = np.array([x, y, theta])

        else:
            # Fill the entire map with random particles
            #TODO: get rid of magic numbers, get map edges in meters from self.map_info
            self.particles[:, 0] = np.random.uniform(-61,25,self.MAX_PARTICLES)
            self.particles[:, 1] = np.random.uniform(-16,48,self.MAX_PARTICLES)
            self.particles[:, 2] = 0

    def precompute_sensor_model(self):
        """
            Fills in a table with precomputed probability values based on range and distance (r, d)
            Range is the laser measurement
            Distance is the location of the obstacle
        """
        print "Sensor model precomputiation: Initializing..."
        table_width = int(self.MAX_RANGE_PX) + 1
        self.sensor_model_table = np.zeros((table_width,table_width))

        for d in range(table_width):
            normalization_coeficient = 0.0
            for r in range(table_width):
                self.sensor_model_table[r, d] = np.exp(-np.power((r - d), 2.0) / (2.0 * np.power(self.sensor_model_sigma, 2.0))) / (self.sensor_model_sigma * np.sqrt(2.0 * np.pi))
                # Range is valid
                if r < d:
                    self.sensor_model_table[r, d] += self.sensor_model_slope * (d - r)
                # Range maxed out
                if r >= self.MAX_RANGE_PX:
                    self.sensor_model_table[r, d] = self.sensor_model_spike

                # Adding a bias to prevent the probability from becoming too small (avoids numerical errors)
                self.sensor_model_table[r, d] += self.sensor_model_bias
                normalization_coeficient += self.sensor_model_table[r, d]
            # Normalize ranges only
            self.sensor_model_table[:,d] /= normalization_coeficient

        self.range_method.set_sensor_model(self.sensor_model_table)

    def motion_model(self, particle_cloud, action):
        """
            Takes an action and applies it to all the particles
            Adds noise
            Tunning parameters are:
                noise standard deviation (X,Y,Theta)
                noise mean location (Could be just in front of the car, currently 0)
        """
        action = np.tile(action, (particle_cloud.shape[0], 1))
        delta = np.empty(action.shape)
        action[:, 0] = action[:, 0]+np.random.normal(loc=self.X_MEAN_POS, scale=np.sqrt(self.X_VARIANCE), size=action.shape[0])
        action[:, 1] = action[:, 1]+np.random.normal(loc=self.Y_MEAN_POS, scale=np.sqrt(self.Y_VARIANCE), size=action.shape[0])
        #TODO: refactor this section to use the 2d rotation matrix function, this notation is confusing :(
        delta[:, 0] = action[:, 0] * np.cos(particle_cloud[:, 2]) + action[:, 1] * np.sin(particle_cloud[:, 2])
        delta[:, 1] = - action[:, 0] * np.sin(particle_cloud[:, 2]) + action[:, 1] * np.cos(particle_cloud[:, 2])
        delta[:, 2] = action[:, 2] + np.random.normal(loc=0, scale=np.sqrt(self.THETA_VARIANCE), size=action.shape[0])
        return particle_cloud + delta

    def sensor_model(self, particle_cloud, obs, weights):
        """
            Updates the weights of the particles based on laser ranges
            Uses RangeLibc magic: https://github.com/kctess5/range_libc/blob/deploy/docs/RangeLibcUsageandInformation.pdf
            Casting ranges and partible positions to float32 is a must. RangeLibc is a cpp implementation and thus, it expects hard types
        """
        if self.first_sensor_update:
            self.queries = np.zeros((particle_cloud.shape[0],3), dtype=np.float32)
            num_queries = self.queries.shape[0]
            self.first_sensor_update = False
            self.ranges = np.zeros(num_queries*len(obs), dtype=np.float32)

        #TODO: Explore custom implementation, wanna try doing the same but in double precession
        self.range_method.calc_range_repeat_angles(particle_cloud.astype(np.float32, order='C'), self.laser_angles.astype(np.float32, order='C'), self.ranges)
        self.range_method.eval_sensor_model(obs.astype(np.float32), self.ranges, weights, self.laser_angles.shape[0], particle_cloud.shape[0])

    def update_particle_filter(self):
        """
            Updates the particle filter
            Steps:
                1. Resample particles based on weight
                2. Calculate average (This is the updated position estimate)
                3. Apply motion model to the particles
                4. Recalculate particle weights (sensor model)
                5. Repeat
        """
        if self.lidar_initialized and self.odom_initialized and self.map_initialized:
            self.particles = self.particles[np.random.choice(self.particles.shape[0], self.MAX_PARTICLES, p=self.weights)]
            self.pose_estimate = np.mean(self.particles, axis=0)
            self.visualize()
            self.particles = self.motion_model(self.particles, self.odomAction)
            self.sensor_model(self.particles, self.laser_ranges, self.weights)
            self.weights /= np.sum(self.weights)
            self.publish_pose_estimate(self.pose_estimate)

### Main function
if __name__=="__main__":
    rospy.init_node("particle_filter")
    ParticleFiler()
    rospy.spin()
