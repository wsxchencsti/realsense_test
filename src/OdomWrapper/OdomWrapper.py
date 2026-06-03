import math
import threading
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


class _OdomSubscriber(Node):
    def __init__(self, topics):
        super().__init__('track_odom_subscriber')

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.lock = threading.Lock()
        self.pose = None
        self.topic = None

        for topic in topics:
            self.create_subscription(Odometry, topic, self.MakeOdomCallback(topic), qos)

    def MakeOdomCallback(self, topic):
        def callback(msg):
            self.OdomCallback(topic, msg)
        return callback

    def OdomCallback(self, topic, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = self.QuaternionToYaw(orientation.x, orientation.y, orientation.z, orientation.w)

        with self.lock:
            self.pose = (position.x, position.y, yaw)
            self.topic = topic

    def QuaternionToYaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


class OdomWrapper:
    def __init__(self, topics=None):
        if not rclpy.ok():
            rclpy.init()

        if topics is None:
            topics = ['/leg_odom2', '/leg_odom']

        self.node = _OdomSubscriber(topics)
        self.break_flag = False
        self.spin_thread = threading.Thread(target=self.SpinThreadFunc)
        self.spin_thread.start()

        print('OdomWrapper topics: {}'.format(', '.join(topics)))

    def __del__(self):
        try:
            self.StopThread()
        except Exception:
            pass

    def StopThread(self):
        if self.break_flag:
            return

        self.break_flag = True
        if self.spin_thread.is_alive():
            self.spin_thread.join()
        self.node.destroy_node()

    def SpinThreadFunc(self):
        while self.break_flag != True:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            time.sleep(0.001)

    def GetPose(self):
        with self.node.lock:
            return self.node.pose

    def GetTopic(self):
        with self.node.lock:
            return self.node.topic
