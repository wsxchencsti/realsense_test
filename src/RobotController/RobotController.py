
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
kMaxControlRadianVelocity = 0.70
kAngularVelocitySmooth = 0.3
kHeadingDeadband = 0.05
kPathMinPointDistance = 0.10
kPathLookaheadDistance = 0.60
kPathTargetDeadband = 0.08
kPathMinForwardDistance = 0.12
kPathMaxHeadingError = 0.65
kPathMinPointsForControl = 3
kLostTargetArriveDistance = 0.20
kLostMaxForwardVelocity = 0.35
kLostMaxHeadingForForward = 0.80
kMaxPathPoints = 1000

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
        self.last_radian_velocity = 0.0
        self.last_person_odom = None
        self.person_path = []

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

    def TransformOdomToRobot(self, odom_point, odom_pose):
        if odom_point is None or odom_pose is None:
            return None

        robot_x, robot_y, robot_yaw = odom_pose
        dx = odom_point[0] - robot_x
        dy = odom_point[1] - robot_y
        target_forward = math.cos(robot_yaw) * dx + math.sin(robot_yaw) * dy
        target_lateral = -math.sin(robot_yaw) * dx + math.cos(robot_yaw) * dy
        return target_forward, target_lateral

    def CalculateRadianVelocity(self, heading_error):
        if heading_error is None or abs(heading_error) < kHeadingDeadband:
            return 0.0

        radian_velocity = heading_error * kAngularHeadingKp
        radian_velocity = max(
            -kMaxControlRadianVelocity,
            min(kMaxControlRadianVelocity, radian_velocity)
        )
        return (
            kAngularVelocitySmooth * self.last_radian_velocity
            + (1.0 - kAngularVelocitySmooth) * radian_velocity
        )

    def GetPathLookaheadTarget(self, odom_pose):
        if odom_pose is None or len(self.person_path) == 0:
            return None, None, None

        robot_x, robot_y, _ = odom_pose
        closest_index = 0
        closest_distance_sq = None
        for index, point in enumerate(self.person_path):
            dx = point[0] - robot_x
            dy = point[1] - robot_y
            distance_sq = dx * dx + dy * dy
            if closest_distance_sq is None or distance_sq < closest_distance_sq:
                closest_distance_sq = distance_sq
                closest_index = index

        target_index = closest_index
        traveled_distance = 0.0
        last_point = self.person_path[closest_index]
        for index in range(closest_index + 1, len(self.person_path)):
            point = self.person_path[index]
            dx = point[0] - last_point[0]
            dy = point[1] - last_point[1]
            traveled_distance += math.sqrt(dx * dx + dy * dy)
            target_index = index
            if traveled_distance >= kPathLookaheadDistance:
                break
            last_point = point

        return self.person_path[target_index], closest_index, target_index

    def ClearPersonPath(self):
        self.person_path = []
        self.last_person_odom = None

    def AddPersonPathPoint(self, person_odom):
        if person_odom is None:
            return False

        if len(self.person_path) == 0:
            self.person_path.append(person_odom)
            return True

        last_x, last_y = self.person_path[-1]
        dx = person_odom[0] - last_x
        dy = person_odom[1] - last_y
        if math.sqrt(dx * dx + dy * dy) < kPathMinPointDistance:
            return False

        self.person_path.append(person_odom)
        if len(self.person_path) > kMaxPathPoints:
            self.person_path.pop(0)
        return True

    def DrawOdomPose(self, frame, odom_pose, y, odom_topic=None, odom_status=None):
        if odom_pose is None:
            receive_count = 0
            if odom_status is not None:
                _, _, receive_count, _ = odom_status
            cv2.putText(frame,"odom invalid recv {}".format(receive_count),(0,y), cv2.FONT_HERSHEY_PLAIN, 1.2, [0,0,255], 1)
            return

        robot_x, robot_y, robot_yaw = odom_pose
        topic_text = odom_topic if odom_topic is not None else "odom"
        qos_text = ""
        if odom_status is not None:
            _, qos_name, _, age = odom_status
            age_text = "--" if age is None else "{:.2f}s".format(age)
            qos_text = " qos {} age {}".format(qos_name, age_text)

        cv2.putText(frame,"odom x {:.2f} y {:.2f} yaw {:.2f}".format(robot_x, robot_y, robot_yaw),
                    (0,y), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
        cv2.putText(frame,"odom {}{}".format(topic_text, qos_text),
                    (0,y + 18), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
    
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
                self.ClearPersonPath()
                self.id_str = ""

            if(self.GetTargetId() != kDefaultTrackId):
                self.SetIsTracking(True)

        else:
            if key == 10 or key == 13 or key == 141:  # 回车
                self.id_str = ""
                self.target_id = 0
                self.ClearPersonPath()
                self.SetIsTracking(False)
        return frame

    def TrackAndDraw(self, frame, box, depth_frame=None, color_intrinsics=None, odom_pose=None, odom_topic=None, odom_status=None):
        shape = frame.shape
        frame = cv2.UMat(frame)

        self.fps_counter.Count()
        frame = cv2.putText(frame, "fps {:.1f}".format(self.fps_counter.GetFps()), (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, [0, 128, 0], 1)
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

            #cal target and radian_velocity
            person_lateral = None
            person_forward = None
            person_odom = None
            person_heading_error = None
            path_target = None
            path_closest_index = None
            path_target_index = None
            path_target_robot = None
            path_heading_error = None
            control_mode = "direct"
            control_forward = None
            control_lateral = None
            heading_error = None
            if person_point is not None:
                person_lateral = -person_point[0]
                person_forward = person_point[2]
                person_odom = self.TransformPersonToOdom(person_forward, person_lateral, odom_pose)
                if person_odom is not None:
                    self.last_person_odom = person_odom
                    self.AddPersonPathPoint(person_odom)
                person_heading_error = math.atan2(person_lateral, person_forward)
                control_forward = person_forward
                control_lateral = person_lateral

            if control_forward is not None and control_lateral is not None:
                heading_error = math.atan2(control_lateral, max(kPathTargetDeadband, control_forward))
                radian_velocity = self.CalculateRadianVelocity(heading_error)
            else:
                radian_velocity = 0.0

            #cal linear_velocity
            distance_error = None
            target_linear_velocity = 0.0
            force_stop_by_distance = person_distance is not None and person_distance <= kTargetDistance
            if person_distance is not None:
                distance_error = person_distance - kTargetDistance
            if force_stop_by_distance or person_distance is None:
                linear_velocity = 0.0
            elif control_forward is not None and control_lateral is not None:
                distance_error = person_distance - kTargetDistance
                if distance_error <= kDistanceDeadband:
                    target_linear_velocity = 0.0
                else:
                    target_linear_velocity = distance_error * kLinearDistanceKp
                    target_linear_velocity = min(kMaxForwardVelocity, target_linear_velocity)

                if heading_error is not None:
                    turn_scale = max(0.2, 1.0 - min(1.0, abs(heading_error)))
                    target_linear_velocity *= turn_scale

                if target_linear_velocity <= 0.0:
                    target_linear_velocity = 0.0
                    linear_velocity = 0.0
                else:
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

            cv2.putText(frame,"ID {:} enter reset".format(self.GetTargetId()),(20,125), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            self.DrawOdomPose(frame, odom_pose, 50, odom_topic, odom_status)
            cv2.putText(frame,"v {:.2f} m/s".format(linear_velocity),(0,250), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            cv2.putText(frame,"w {:.2f} rad/s".format(radian_velocity),(0,270), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            if person_distance is not None:
                cv2.putText(frame,"depth {:.2f} target {:.2f}".format(person_distance, kTargetDistance),(0,290), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
                cv2.putText(frame,"mode {} err {:.2f} target_v {:.2f}".format(control_mode, distance_error, target_linear_velocity),(0,310), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
                if person_point is not None:
                    cv2.putText(frame,"rel x {:.2f} z {:.2f} head {:.2f}".format(person_lateral, person_forward, person_heading_error),(0,330), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
                    if person_odom is not None:
                        cv2.putText(frame,"person odom x {:.2f} y {:.2f}".format(person_odom[0], person_odom[1]),(0,350), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
                        cv2.putText(frame,"path n {:d} last x {:.2f} y {:.2f}".format(len(self.person_path), person_odom[0], person_odom[1]),(0,370), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
                if control_forward is not None and heading_error is not None:
                    cv2.putText(frame,"ctrl x {:.2f} z {:.2f} head {:.2f}".format(control_lateral, control_forward, heading_error),(0,390), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            else:
                cv2.putText(frame,"depth invalid",(0,290), cv2.FONT_HERSHEY_PLAIN, 1.2, [0,0,255], 1)

            #pub cmdvel
            if kUseRos1Transfer:
                self.ros1_transfer.SendCmdVel(linear_velocity, radian_velocity)
            else:
                self.ros2_transfer.SendCmdVel(linear_velocity, radian_velocity)
            self.last_linear_velocity = linear_velocity
            self.last_radian_velocity = radian_velocity
            return frame
        else:
            control_mode = "lost_stop"
            linear_velocity = 0.0
            radian_velocity = 0.0
            target_linear_velocity = 0.0
            lost_target_robot = self.TransformOdomToRobot(self.last_person_odom, odom_pose)
            lost_target_distance = None
            lost_heading_error = None
            if lost_target_robot is not None:
                lost_forward, lost_lateral = lost_target_robot
                lost_target_distance = math.sqrt(lost_forward * lost_forward + lost_lateral * lost_lateral)
                lost_heading_error = math.atan2(lost_lateral, lost_forward)
                if lost_target_distance > kLostTargetArriveDistance:
                    control_mode = "lost_last"
                    radian_velocity = self.CalculateRadianVelocity(lost_heading_error)
                    if abs(lost_heading_error) <= kLostMaxHeadingForForward and lost_forward > 0:
                        target_linear_velocity = lost_target_distance * kLinearDistanceKp
                        target_linear_velocity = min(kLostMaxForwardVelocity, target_linear_velocity)
                        turn_scale = max(0.2, 1.0 - min(1.0, abs(lost_heading_error)))
                        target_linear_velocity *= turn_scale
                        linear_velocity = (
                            kLinearVelocitySmooth * self.last_linear_velocity
                            + (1.0 - kLinearVelocitySmooth) * target_linear_velocity
                        )

            if kUseRos1Transfer:
                self.ros1_transfer.SendCmdVel(linear_velocity, radian_velocity)
            else:
                self.ros2_transfer.SendCmdVel(linear_velocity, radian_velocity)
            self.last_linear_velocity = linear_velocity
            self.last_radian_velocity = radian_velocity
            self.DrawOdomPose(frame, odom_pose, 50, odom_topic, odom_status)
            cv2.putText(frame,"lost ID {:}".format(self.GetTargetId()),(20,100), cv2.FONT_HERSHEY_PLAIN, 1.2, [0,0,255], 1)
            cv2.putText(frame,"enter reset",(20,120), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            cv2.putText(frame,"v {:.2f} m/s".format(linear_velocity),(0,250), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            cv2.putText(frame,"w {:.2f} rad/s".format(radian_velocity),(0,270), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            cv2.putText(frame,"mode {} target_v {:.2f}".format(control_mode, target_linear_velocity),(0,290), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            if lost_target_distance is not None:
                cv2.putText(frame,"last dist {:.2f} head {:.2f}".format(lost_target_distance, lost_heading_error),(0,310), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            if self.last_person_odom is not None:
                cv2.putText(frame,"last odom x {:.2f} y {:.2f}".format(self.last_person_odom[0], self.last_person_odom[1]),(0,330), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
            return frame

    def NonTrackAndDraw(self, frame, odom_pose=None, odom_topic=None, odom_status=None):
        frame = cv2.UMat(frame)
        self.fps_counter.Count()
        frame = cv2.putText(frame, "fps {:.1f}".format(self.fps_counter.GetFps()), (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, [0, 128, 0], 1)
        self.InputAndProcess(frame)
        if kUseRos1Transfer:
            self.ros1_transfer.SendCmdVel(0.0, 0.0)
        else:
            self.ros2_transfer.SendCmdVel(0.0, 0.0)
        self.last_linear_velocity = 0.0
        self.last_radian_velocity = 0.0
        self.DrawOdomPose(frame, odom_pose, 50, odom_topic, odom_status)
        cv2.putText(frame,"stop",(20,100), cv2.FONT_HERSHEY_PLAIN, 1.2, [0,0,255], 1)
        cv2.putText(frame,"input ID:",(20,120), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
        cv2.putText(frame,"v 0.00 m/s",(0,250), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
        cv2.putText(frame,"w 0.00 rad/s",(0,270), cv2.FONT_HERSHEY_PLAIN, 1.2, [255,0,0], 1)
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
        if(self.GetIsTracking()):
            return self.TrackAndDraw(frame, None, depth_frame, color_intrinsics, odom_pose, odom_topic, odom_status)
        return self.NonTrackAndDraw(frame, odom_pose, odom_topic, odom_status)
