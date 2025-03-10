#------------------------------------------------------------------------------
# This script receives video frames and spatial input data from the HoloLens.
# The received head pointer, hand joint positions and gaze pointer are 
# projected onto the video frame.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import multiprocessing as mp
import numpy as np
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_utilities
import hl2ss_mp
import hl2ss_3dcv
import hl2ss_sa
from pymemcache.client import base
from ultralytics import YOLO

# Settings --------------------------------------------------------------------

# HoloLens 2 address
host = "192.168.10.149"
mem_client = base.Client(('localhost', 11211))

# Camera parameters
# See etc/hl2_capture_formats.txt for a list of supported formats
pv_width     = 760
pv_height    = 428
pv_framerate = 30

# Marker properties
radius = 5
head_color  = (  0,   0, 255)
left_color  = (  0, 255,   0)
right_color = (255,   0,   0)
gaze_color  = (255,   0, 255)
thickness = -1

# Buffer length in seconds
buffer_length = 5

# Spatial Mapping settings
triangles_per_cubic_meter = 1000
mesh_threads = 2
sphere_center = [0, 0, 0]
sphere_radius = 5

collect_data = False
run_detector = True

obj_class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
filtered_class_names = ['tv', 'laptop', 'cell phone', 'book', 'cup', 'bottle', 'keyboard']
#------------------------------------------------------------------------------

def get_closest_bbox(bboxes, classes, m, c):
    closest_distance = float('inf') 
    closest_bbox = None
    for i,(bbox,cls) in enumerate(zip(bboxes, classes)):
        if cls == 'person':
            continue
        if m is None or c is None:
            break
        else:
            x = bbox[0] + bbox[2]/2
            y = bbox[1] + bbox[3]/2
            distance = abs(y - (m * x + c)) / (m**2 + 1)**0.5
            if distance < closest_distance:
                closest_distance = distance
                closest_bbox = i
    return closest_bbox

# def rule_based_grounding(bboxes, classes, m):
#     pass

if __name__ == '__main__':
    # Keyboard events ---------------------------------------------------------
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Start Spatial Mapping data manager --------------------------------------
    # Set region of 3D space to sample
    volumes = hl2ss.sm_bounding_volume()
    volumes.add_sphere(sphere_center, sphere_radius)

    # Download observed surfaces
    sm_manager = hl2ss_sa.sm_manager(host, triangles_per_cubic_meter, mesh_threads)
    sm_manager.open()
    sm_manager.set_volumes(volumes)
    sm_manager.get_observed_surfaces()

    #load yolo model
    yolo = None
    if run_detector:
        yolo = YOLO("yolov8n.pt")
    
    # Start PV and Spatial Input streams --------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate))
    producer.configure(hl2ss.StreamPort.SPATIAL_INPUT, hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT))
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_framerate * buffer_length)
    producer.initialize(hl2ss.StreamPort.SPATIAL_INPUT, hl2ss.Parameters_SI.SAMPLE_RATE * buffer_length)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.start(hl2ss.StreamPort.SPATIAL_INPUT)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, ...)
    sink_si = consumer.create_sink(producer, hl2ss.StreamPort.SPATIAL_INPUT, manager, None)
    sink_pv.get_attach_response()
    sink_si.get_attach_response()

    idx = 0
    results = None
    bboxes = None
    classes = None
    detect_response = ""

    # Main Loop ---------------------------------------------------------------
    while (enable):
        # Download observed surfaces ------------------------------------------
        filtered_bboxes = []
        filtered_classes = []
        sm_manager.get_observed_surfaces()

        # Wait for PV frame ---------------------------------------------------
        sink_pv.acquire()

        # Get PV frame and nearest (in time) Spatial Input frame --------------
        _, data_pv = sink_pv.get_most_recent_frame()
        if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
            continue

        _, data_si = sink_si.get_nearest(data_pv.timestamp)
        if (data_si is None):
            continue

        image = data_pv.payload.image
        if mem_client.get('state') == b'1':
            cv2.imwrite("/home/user/Projects/LLM/images/img1.jpg", image)
        
        if collect_data:
            cv2.imwrite("/home/user/Projects/detector/data/img" + str(idx) + ".jpg", image)
        si = hl2ss.unpack_si(data_si.payload)

        # Update PV intrinsics ------------------------------------------------
        # PV intrinsics may change between frames due to autofocus
        pv_intrinsics = hl2ss.create_pv_intrinsics(data_pv.payload.focal_length, data_pv.payload.principal_point)
        pv_extrinsics = np.eye(4, 4, dtype=np.float32)
        pv_intrinsics, pv_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

        # Compute world to PV image transformation matrix ---------------------
        world_to_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(pv_extrinsics) @ hl2ss_3dcv.camera_to_image(pv_intrinsics)

        #run detector
        if run_detector:
            results = yolo.track(image, persist=True)
            image = results[0].plot()
            # cv2.imwrite("/home/user/Projects/LLM/images/test.jpg", image)
            classes = [obj_class_names[int(x)] for x in list(results[0].boxes.cls.cpu().numpy())]
            bboxes  = results[0].boxes.xyxy.cpu().numpy()

            # filter bboxes and classes with filtered_classes

            # for i,(bbox,cls) in enumerate(zip(bboxes, classes)):
            #     if cls in filtered_class_names:
            #         filtered_bboxes.append(bbox)
            #         filtered_classes.append(cls)
            valid_cls = [k for k, cls in enumerate(classes) if cls in filtered_class_names]
            bboxes = bboxes[valid_cls]
            classes = [classes[i] for i in valid_cls]
            # mem_client.set('detector', )

            # print(int(results[0].boxes.cls.cpu().numpy()))
            print('........................')
            # detect_response = '\nxyxy: ' + str(results[0].boxes.xyxy) + '\nCLASSES: ' + str(classes)
            detect_response = '\nxyxy: ' + str(results[0].boxes.xyxy[valid_cls]) + '\nCLASSES: ' + str(classes)
            # mem_client.set('detector', detect_response)
            # print(detect_response)
            print('....................')

        # Draw Head Pointer ---------------------------------------------------
        if (si.is_valid_head_pose()):
            head_pose = si.get_head_pose()
            head_ray = hl2ss_utilities.si_ray_to_vector(head_pose.position, head_pose.forward)
            d = sm_manager.cast_rays(head_ray)
            if (np.isfinite(d)):
                head_point = hl2ss_utilities.si_ray_to_point(head_ray, d)
                head_image_point = hl2ss_3dcv.project(head_point, world_to_image)
                hl2ss_utilities.draw_points(image, head_image_point.astype(np.int32), radius, head_color, thickness)


        # Draw Left Hand joints -----------------------------------------------
        if (si.is_valid_hand_left()):
            left_hand = si.get_hand_left()
            left_joints = hl2ss_utilities.si_unpack_hand(left_hand)
            left_image_points = hl2ss_3dcv.project(left_joints.positions, world_to_image)
            #hl2ss_utilities.draw_points(image, left_image_points.astype(np.int32), radius, left_color, thickness)
            X_train = []
            Y_train = []

            # draw points on image using opencv
            for i,(x,y) in enumerate(left_image_points.astype(np.int32)):
                cv2.circle(image, (x,y), 5, right_color, -1)

                if i>=6 and i<11:
                    X_train.append(x)
                    Y_train.append(y)

            m,c = np.polyfit(X_train, Y_train, 1)

            width = image.shape[1]
            x1 = 0
            y1 = int(c)
            x2 = width
            y2 = int(m*width + c)
            cv2.line(image, (x1, y1), (x2, y2), left_color, 2)

        # Draw Right Hand joints ----------------------------------------------
        if (si.is_valid_hand_right()):
            right_hand = si.get_hand_right()
            right_joints = hl2ss_utilities.si_unpack_hand(right_hand)
            right_image_points = hl2ss_3dcv.project(right_joints.positions, world_to_image)
            # hl2ss_utilities.draw_points(image, right_image_points.astype(np.int32), radius, right_color, thickness)

            X_train = []
            Y_train = []

            # draw points on image using opencv
            for i,(x,y) in enumerate(right_image_points.astype(np.int32)):
                cv2.circle(image, (x,y), 5, right_color, -1)

                if i>=6 and i<11:
                    X_train.append(x)
                    Y_train.append(y)

            m,c = np.polyfit(X_train, Y_train, 1)

            width = image.shape[1]
            x1 = 0
            y1 = int(c)
            x2 = width
            y2 = int(m*width + c)
            cv2.line(image, (x1, y1), (x2, y2), right_color, 2)

            # pointed = get_closest_bbox(bboxes, classes, m, c)
            pointed = get_closest_bbox(bboxes, classes, m, c)
            # detect_response += '\nPointed: ' + str(pointed)
            mem_client.set('detector', detect_response)
            print(detect_response)



        # Draw Gaze Pointer ---------------------------------------------------
        if (si.is_valid_eye_ray()):
            eye_ray = si.get_eye_ray()
            eye_ray_vector = hl2ss_utilities.si_ray_to_vector(eye_ray.origin, eye_ray.direction)
            d = sm_manager.cast_rays(eye_ray_vector)
            if (np.isfinite(d)):
                gaze_point = hl2ss_utilities.si_ray_to_point(eye_ray_vector, d)
                gaze_image_point = hl2ss_3dcv.project(gaze_point, world_to_image)
                hl2ss_utilities.draw_points(image, gaze_image_point.astype(np.int32), radius, gaze_color, thickness)
                
        # Display frame -------------------------------------------------------
        cv2.imshow('Video', image)
        cv2.waitKey(1)
        if collect_data:
            idx += 1
        
    # Stop Spatial Mapping data manager ---------------------------------------
    sm_manager.close()

    # Stop PV and Spatial Input streams ---------------------------------------
    sink_pv.detach()
    sink_si.detach()
    producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.stop(hl2ss.StreamPort.SPATIAL_INPUT)

    # Stop PV subsystem -------------------------------------------------------
    hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Stop keyboard events ----------------------------------------------------
    listener.join()
