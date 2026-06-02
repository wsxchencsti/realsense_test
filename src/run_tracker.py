
import cv2
import signal

from RealSenseWrapper import RealSenseWrapper
from RobotController import RobotController

break_flag = False
def set_break_flag(signum, frame):
    print("set_break_flag")
    global break_flag
    break_flag = True

signal.signal(signal.SIGINT, set_break_flag)

realsense_wrapper = RealSenseWrapper.RealSenseWrapper()
robot_controller = RobotController.RobotController()

while break_flag != True:
    ############
    # get frame
    ############
    bgr_frame = realsense_wrapper.GetFrame()
    if type(bgr_frame) == type(None):
        continue
    depth_frame = realsense_wrapper.GetDepthFrame()

    ####################
    # control the robot
    ####################
    final_frame = robot_controller.Run(bgr_frame)
    if type(final_frame) == type(None):
        continue
    if depth_frame is not None:
        center_x = bgr_frame.shape[1] // 2
        center_y = bgr_frame.shape[0] // 2
        center_distance = depth_frame.get_distance(center_x, center_y)
        cv2.putText(final_frame, "center depth {:.02f} m".format(center_distance), (0, 350),
                    cv2.FONT_HERSHEY_PLAIN, 2, [255, 0, 0], 3)

    #################
    # show final img
    #################
    cv2.imshow("DR People Tracking", final_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

realsense_wrapper.StopThread()

cv2.destroyAllWindows()
