import cv2 as cv
import numpy as np

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth = 368
inHeight = 368

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

frame = cv.imread("media.jpeg")
if frame is None:
    print("Error: Could not read image")
    exit()

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
out = net.forward()
out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

assert(len(BODY_PARTS) == out.shape[1])

points = []
for i in range(len(BODY_PARTS)):
    # Slice heatmap of corresponding body part.
    heatMap = out[0, i, :, :]

    # Originally, we try to find all the local maximums. To simplify a sample
    # we just find a global one. However only a single pose at the same time
    # could be detected this way.
    _, conf, _, point = cv.minMaxLoc(heatMap)
    x = (frameWidth * point[0]) / out.shape[3]
    y = (frameHeight * point[1]) / out.shape[2]
    # Add a point if its confidence is higher than the threshold.
    points.append((int(x), int(y)) if conf > 0.2 else None)

lengths = {}
for pair in POSE_PAIRS:
    partFrom = pair[0]
    partTo = pair[1]
    assert(partFrom in BODY_PARTS)
    assert(partTo in BODY_PARTS)

    idFrom = BODY_PARTS[partFrom]
    idTo = BODY_PARTS[partTo]

    if points[idFrom] and points[idTo]:
        length = np.sqrt((points[idTo][0] - points[idFrom][0]) ** 2 + (points[idTo][1] - points[idFrom][1]) ** 2)
        lengths[f"{partFrom}-{partTo}"] = length

# Convert lengths to feet
pixel_to_feet_ratio_width = 4.1222 / frameWidth  # Ratio of width in pixels to width in feet
pixel_to_feet_ratio_height = 6 / frameHeight  # Ratio of height in pixels to height in feet

for key, value in lengths.items():
    lengths[key] = value * pixel_to_feet_ratio_width if 'Width' in key else value * pixel_to_feet_ratio_height

print("Lengths of lines connecting keypoints in feet:")
for key, value in lengths.items():
    print(f"{key}: {value} feet")
