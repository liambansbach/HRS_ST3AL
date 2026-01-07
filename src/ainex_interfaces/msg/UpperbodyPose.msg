std_msgs/Header header

# Normalized coordinates (MediaPipe pose landmarks):
# x,y in [0..1] relative to image width/height
# z is MediaPipe relative depth

geometry_msgs/Point left_shoulder
geometry_msgs/Point left_elbow
geometry_msgs/Point left_wrist
geometry_msgs/Point left_pinky
geometry_msgs/Point left_index

geometry_msgs/Point right_shoulder
geometry_msgs/Point right_elbow
geometry_msgs/Point right_wrist
geometry_msgs/Point right_pinky
geometry_msgs/Point right_index

# Confidence (MediaPipe "visibility" if present; otherwise publish 1.0)
float32 left_shoulder_vis
float32 left_elbow_vis
float32 left_wrist_vis
float32 left_pinky_vis
float32 left_index_vis

float32 right_shoulder_vis
float32 right_elbow_vis
float32 right_wrist_vis
float32 right_pinky_vis
float32 right_index_vis
