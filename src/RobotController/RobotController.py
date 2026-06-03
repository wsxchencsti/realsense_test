
import cv2
import math
import numpy as np

from . FpsCounter import FpsCounter
from . YoloWrapper import YoloWrapper
from . ROSTransfer import TransferConstants

kUseRos1Transfer = False

if kUseRos1Transfer:
    from . ROSTransfer import ROS1Transfer
else:
    from . ROSTransfer import ROS2Transfer

kDefaultTrackId = 0
kDefaultTrackMode = False
kTargetDistance = 0.5
kDistanceDeadband = 0.05
kLinearDistanceKp = 0.6
kMaxForwardVelocity = 0.6
kLinearVelocitySmooth = 0.6
kAngularHeadingKp = 1.2
kHeadingDeadband = 0.05

class RobotController(object):
    def __init__(self):
        self.fps_counter = FpsCounter.FpsCounter()
        self.yolo_wrapper = YoloWrapper.YoloWrapper()
        if kUseRos1Transfer:
            self.ros1_transfer = ROS1Transfer.ROS1Transfer()
        else:
            self.ros2_transfer = ROS2Transfer.ROS2Transfer()
    
        self.is_tracking = kDefaultTrackMode
        self.target_id = kDefaultTrackId
        self.id_str = ""

        self.last_linear_velocity = 0.0

    def SetTargetId(self, id):
        self.target_id = id

    def GetTargetId(self):
        return self.target_id

    def SetIsTracking(self, state):
        self.is_tracking = state

    def GetIsTracking(self):
        return self.is_tracking

    def FindTarget(self, boxes):
        for box in boxes:
            if box.id != None:
                if box.id.item() == self.GetTargetId():
                    return box

    def GetBoxCenterDistance(self, depth_frame, x1, y1, x2, y2, image_width, image_height):
        if depth_frame is None:
            return None, None

        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(0, min(image_width - 1, x2))
        y2 = max(0, min(image_height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None, None

        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        half_width = max(3, box_width // 8)
        half_height = max(3, box_height // 8)

        rx1 = max(0, center_x - half_width)
        ry1 = max(0, center_y - half_height)
        rx2 = min(image_width - 1, center_x + half_width)
        ry2 = min(image_height - 1, center_y + half_height)

        distances = []
        step_x = max(1, (rx2 - rx1) // 8)
        step_y = max(1, (ry2 - ry1) // 8)

        for y in range(ry1, ry2 + 1, step_y):
            for x in range(rx1, rx2 + 1, step_x):
                distance = depth_frame.get_distance(x, y)
                if distance > 0:
                    distances.append(distance)

        if len(distances) == 0:
            return None, (rx1, ry1, rx2, ry2)

        return float(np.median(distances)), (rx1, ry1, rx2, ry2)

    def DeprojectPixelToPoint(self, intrinsics, pixel_x, pixel_y, depth):
        if intrinsics is None or depth is None or depth <= 0:
            return None

        camera_x = (pixel_x - intrinsics.ppx) / intrinsics.fx * depth
        camera_y = (pixel_y - intrinsics.ppy) / intrinsics.fy * depth
        camera_z = depth
        return camera_x, camera_y, camera_z

    def TransformPersonToOdom(self, person_forward, person_lateral, odom_pose):
        if person_forward is None or person_lateral is None or odom_pose is None:
            return None

        robot_x, robot_y, robot_yaw = odom_pose
        person_odom_x = robot_x + math.cos(robot_yaw) * person_forward - math.sin(robot_yaw) * person_lateral
        person_odom_y = robot_y + math.sin(robot_yaw) * person_forward + math.cos(robot_yaw) * person_lateral
        return person_odom_x, person_odom_y

    def FormatValue(self, value, unit="", precision=2):
        if value is None:
            return "--"
        return ("{:.%df}{}" % precision).format(value, unit)

    def DrawDebugPanel(self, frame, lines, x=10, y=10):
        line_height = 22
        padding = 8
        width = 500
        height = padding * 2 + line_height * len(lines)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), [20, 20, 20], -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for index, (text, color) in enumerate(lines):
            text_y = y + padding + 16 + index * line_height
            cv2.putText(frame, text, (x + padding, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def BuildOdomLines(self, odom_pose, odom_topic=None, odom_status=None):
        if odom_pose is None:
            receive_count = 0
            if odom_status is not None:
                _, _, receive_count, _ = odom_status
            return [("ODOM invalid  recv={}".format(receive_count), [0, 80, 255])]

        robot_x, robot_y, robot_yaw = odom_pose
        topic_text = odom_topic if odom_topic is not None else "odom"
        qos_text = ""
        if odom_status is not None:
            _, qos_name, _, age = odom_status
            age_text = "--" if age is None else "{:.2f}s".format(age)
            qos_text = "  qos={} age={}".format(qos_name, age_text)

        return [
            ("ODOM x={:.2f} y={:.2f} yaw={:.2f}".format(robot_x, robot_y, robot_yaw), [120, 220, 255]),
            ("ODOM src={}{}".format(topic_text, qos_text), [120, 220, 255]),
        ]
    
    def InputAndProcess(self, frame):
        key = cv2.waitKey(1)
        if(self.GetIsTracking() == False):
            if key >= ord('0') and key <= ord('9'):  # 大键盘输入
                self.id_str += str((int(key - ord('0'))))
            elif key == ord('\b'):  # 退格
                self.id_str = self.id_str[:-1]
            elif key == 10 or key == 13 or key == 141:  # 回车
                if(self.id_str == ""):
                    self.SetTargetId(0)
                else:
                    self.SetTargetId(int(self.id_str))
                self.id_str = ""

            if(self.GetTargetId() != kDefaultTrackId):
                self.SetIsTracking(True)

        else:
            if key == 10 or key == 13 or key == 141:  # 回车
                self.id_str == ""
                self.target_id = 0
                self.SetIsTracking(False)
        return frame

    def TrackAndDraw(self, frame, box, depth_frame=None, color_intrinsics=None, odom_pose=None, odom_topic=None, odom_status=None):
        shape = frame.shape

        self.fps_counter.Count()
        self.InputAndProcess(frame)

        if(box != None):            
            x1 = int(box.xyxy[0][0].item())
            y1 = int(box.xyxy[0][1].item())
            x2 = int(box.xyxy[0][2].item())
            y2 = int(box.xyxy[0][3].item())
            center=((x1+x2)//2, (y1+y2)//2)
            person_distance, depth_region = self.GetBoxCenterDistance(
                depth_frame, x1, y1, x2, y2, shape[1], shape[0])
            person_point = self.DeprojectPixelToPoint(color_intrinsics, center[0], center[1], person_distance)
            cv2.circle(frame,center,2,[0,0,255],-1) # 画出选框中心点

            #cal radian_velocity
            person_lateral = None
            person_forward = None
            person_odom = None
            heading_error = None
            if person_point is not None:
                person_lateral = -person_point[0]
                person_forward = person_point[2]
                person_odom = self.TransformPersonToOdom(person_forward, person_lateral, odom_pose)
                heading_error = math.atan2(person_lateral, person_forward)
                if abs(heading_error) < kHeadingDeadband:
                    radian_velocity = 0.0
                else:
                    radian_velocity = heading_error * kAngularHeadingKp
                    radian_velocity = max(
                        -TransferConstants.kMaxRadianVelocity,
                        min(TransferConstants.kMaxRadianVelocity, radian_velocity)
                    )
            else:
                radian_velocity = 0.0

            #cal linear_velocity
            distance_error = None
            target_linear_velocity = 0.0
            if person_distance is not None:
                distance_error = person_distance - kTargetDistance
                if distance_error <= kDistanceDeadband:
                    target_linear_velocity = 0.0
                else:
                    target_linear_velocity = distance_error * kLinearDistanceKp
                    target_linear_velocity = min(kMaxForwardVelocity, target_linear_velocity)
                linear_velocity = (
                    kLinearVelocitySmooth * self.last_linear_velocity
                    + (1.0 - kLinearVelocitySmooth) * target_linear_velocity
                )
            else:
                linear_velocity = 0.0

            #draw
            label = '{}{:d}'.format("", self.GetTargetId())
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), [255,128,128], 2)
            if depth_region is not None:
                rx1, ry1, rx2, ry2 = depth_region
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), [0,255,0], 2)
            cv2.rectangle(frame,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), [255,128,128],-1)
            cv2.putText(frame,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)

            debug_lines = [
                ("FPS {:.1f}  MODE tracking  ID {}  enter=reset".format(self.fps_counter.GetFps(), self.GetTargetId()), [80, 255, 80]),
                ("CMD v={}m/s  w={}rad/s  target_v={}m/s".format(
                    self.FormatValue(linear_velocity), self.FormatValue(radian_velocity), self.FormatValue(target_linear_velocity)), [255, 180, 80]),
                ("DEPTH person={}m  target={}m  err={}m".format(
                    self.FormatValue(person_distance), self.FormatValue(kTargetDistance), self.FormatValue(distance_error)), [255, 180, 80]),
                ("REL lateral={}m  forward={}m  heading={}rad".format(
                    self.FormatValue(person_lateral), self.FormatValue(person_forward), self.FormatValue(heading_error)), [255, 220, 120]),
            ]
            if person_odom is not None:
                debug_lines.append(("PERSON_ODOM x={:.2f} y={:.2f}".format(person_odom[0], person_odom[1]), [120, 220, 255]))
            else:
                debug_lines.append(("PERSON_ODOM --", [120, 120, 120]))
            debug_lines.extend(self.BuildOdomLines(odom_pose, odom_topic, odom_status))
            self.DrawDebugPanel(frame, debug_lines)

            #pub cmdvel
            if kUseRos1Transfer:
                self.ros1_transfer.SendCmdVel(linear_velocity, radian_velocity)
            else:
                self.ros2_transfer.SendCmdVel(linear_velocity, radian_velocity)
            self.last_linear_velocity = linear_velocity
            return frame
        else:
            if kUseRos1Transfer:
                self.ros1_transfer.SendCmdVel(0.0, 0.0)
            else:
                self.ros2_transfer.SendCmdVel(0.0, 0.0)
            self.last_linear_velocity = 0.0
            debug_lines = [
                ("FPS {:.1f}  MODE lost  ID {}  enter=reset".format(self.fps_counter.GetFps(), self.GetTargetId()), [0, 80, 255]),
                ("CMD v=0.00m/s  w=0.00rad/s", [255, 180, 80]),
                ("TARGET not visible", [0, 80, 255]),
            ]
            debug_lines.extend(self.BuildOdomLines(odom_pose, odom_topic, odom_status))
            self.DrawDebugPanel(frame, debug_lines)
            return frame

    def NonTrackAndDraw(self, frame, odom_pose=None, odom_topic=None, odom_status=None):
        self.fps_counter.Count()
        self.InputAndProcess(frame)
        if kUseRos1Transfer:
            self.ros1_transfer.SendCmdVel(0.0, 0.0)
        else:
            self.ros2_transfer.SendCmdVel(0.0, 0.0)
        self.last_linear_velocity = 0.0
        debug_lines = [
            ("FPS {:.1f}  MODE idle".format(self.fps_counter.GetFps()), [80, 255, 80]),
            ("CMD v=0.00m/s  w=0.00rad/s", [255, 180, 80]),
            ("INPUT target ID: {}".format(self.id_str if self.id_str != "" else "--"), [255, 255, 255]),
        ]
        debug_lines.extend(self.BuildOdomLines(odom_pose, odom_topic, odom_status))
        self.DrawDebugPanel(frame, debug_lines)
        return frame

    def Run(self, frame, depth_frame=None, color_intrinsics=None, odom_pose=None, odom_topic=None, odom_status=None):
        results = self.yolo_wrapper.Track(frame)
        if(len(results)>0):
            if(self.GetIsTracking()):
                box = self.FindTarget(results[0].boxes)                
                return self.TrackAndDraw(frame, box, depth_frame, color_intrinsics, odom_pose, odom_topic, odom_status)
            else:
                frame = results[0].plot()
                return self.NonTrackAndDraw(frame, odom_pose, odom_topic, odom_status)
