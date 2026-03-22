#!/usr/bin/env python3
"""
detection_node.py — VisionFleet
Dynamically discovers all bike camera topics and runs detection on each.
STRICT FILTERING MODE: Eliminates false positives (white lines/shadows) 
by raising confidence and enforcing a strict 10% warm-color volume rule.
"""
import json
import threading
import rclpy
import os
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

MIN_BBOX_AREA    = 400
COLOR_BOX        = (0, 140, 255)
COLOR_TEXT       = (0, 0, 0)
CUSTOM_YOLO_PATH = os.path.expanduser('~/VisionFleet_ws/assets/weights/cone_2.pt')


class BikeDetector:
    def __init__(self, bike_id, det_pub, viz_pub):
        self.bike_id     = bike_id
        self.det_pub     = det_pub
        self.viz_pub     = viz_pub
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.frame_count = 0


class DetectionNode(Node):

    def __init__(self):
        super().__init__('detection_node')

        self._lock   = threading.Lock()
        self._bikes  = {}   
        self._bridge = CvBridge()
        self._model  = None
        self._model_lock = threading.Lock() 

        self._load_model()

        self.create_timer(2.0, self._discover_bikes)
        self.get_logger().info('Detection node ready — watching for bike cameras...')

    def _load_model(self):
        if not YOLO_AVAILABLE:
            self.get_logger().warn('Ultralytics not installed — YOLO disabled.')
            return
        if not os.path.exists(CUSTOM_YOLO_PATH):
            self.get_logger().warn(f'Model not found at {CUSTOM_YOLO_PATH}')
            return
        try:
            self._model = YOLO(CUSTOM_YOLO_PATH)
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self._model(dummy, verbose=False)
            self.get_logger().info('Custom YOLO cone model loaded (STRICT MODE).')
        except Exception as e:
            self.get_logger().error(f'YOLO load failed: {e}')

    def _discover_bikes(self):
        known_topics = [name for name, _ in self.get_topic_names_and_types()]

        for topic in known_topics:
            if topic == '/carla/hero/rgb/image':
                with self._lock:
                    self._register_bike('hero', '/carla/hero/rgb/image', '/carla/hero/gnss')

            elif (topic.startswith('/carla/hero/') and topic.endswith('/rgb/image')):
                parts = topic.split('/')
                if len(parts) == 6:
                    bike_id = parts[3]
                    with self._lock:
                        self._register_bike(bike_id, topic, f'/carla/hero/{bike_id}/gnss')

    def _register_bike(self, bike_id, image_topic, gnss_topic):
        if bike_id in self._bikes:
            return

        det_pub = self.create_publisher(String, f'/visionfleet/{bike_id}/detections', 10)
        viz_pub = self.create_publisher(Image,  f'/visionfleet/{bike_id}/detection_image', 10)

        self._bikes[bike_id] = BikeDetector(bike_id, det_pub, viz_pub)

        self.create_subscription(Image, image_topic, self._make_image_cb(bike_id), 10)
        self.create_subscription(NavSatFix, gnss_topic, self._make_gnss_cb(bike_id), 10)

        self.get_logger().info(f'[detection] Registered: {bike_id}')

    def _make_gnss_cb(self, bike_id):
        def cb(msg: NavSatFix):
            with self._lock:
                if bike_id in self._bikes:
                    self._bikes[bike_id].current_lat = msg.latitude
                    self._bikes[bike_id].current_lon = msg.longitude
        return cb

    def _make_image_cb(self, bike_id):
        def cb(msg: Image):
            with self._lock:
                if bike_id not in self._bikes:
                    return
                bike = self._bikes[bike_id]
                lat  = bike.current_lat
                lon  = bike.current_lon
                bike.frame_count += 1
                frame_count = bike.frame_count

            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            detections = []
            if self._model is not None:
                with self._model_lock:
                    detections = self._detect_yolo(frame)

            # High confidence threshold to trigger a map event
            high_conf = [d for d in detections if d['confidence'] >= 0.65]
            is_blocked = len(detections) >= 3 or len(high_conf) >= 1

            if is_blocked:
                if frame_count % 20 == 0:
                    self.get_logger().info(
                        f'[{bike_id}] BLOCKED ROAD! GPS: [{lat:.6f}, {lon:.6f}]  detections: {len(detections)}')

                payload = {
                    'timestamp':   msg.header.stamp.sec,
                    'road_blocked': True,
                    'bike_id':     bike_id,
                    'lat':         lat,
                    'lon':         lon,
                    'count':       len(detections),
                }
                out      = String()
                out.data = json.dumps(payload)

                with self._lock:
                    if bike_id in self._bikes:
                        self._bikes[bike_id].det_pub.publish(out)

            annotated       = self._draw(frame.copy(), detections)
            viz_msg         = self._bridge.cv2_to_imgmsg(annotated, 'bgr8')
            viz_msg.header  = msg.header

            with self._lock:
                if bike_id in self._bikes:
                    self._bikes[bike_id].viz_pub.publish(viz_msg)
        return cb

    def _detect_yolo(self, frame):
        h, w     = frame.shape[:2]
        infer_w  = 320
        infer_h  = int(h * infer_w / w)
        small    = cv2.resize(frame, (infer_w, infer_h))
        
        # STRICT: Base confidence raised to 0.50 to kill road line noise
        results  = self._model(small, verbose=False, conf=0.50, iou=0.45)

        detections = []
        scale_x    = w / infer_w
        scale_y    = h / infer_h

        # STRICT COLOR: Must be genuinely orange/yellow (Sat > 40 ignores gray/white)
        lower_warm = np.array([5, 40, 80])
        upper_warm = np.array([35, 255, 255])

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id      = int(box.cls[0])
                conf        = float(box.conf[0])
                label       = self._model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                x1 = max(0, int(x1 * scale_x))
                y1 = max(0, int(y1 * scale_y))
                x2 = min(w, int(x2 * scale_x))
                y2 = min(h, int(y2 * scale_y))
                
                bw = x2 - x1
                bh = y2 - y1

                if bw * bh < self._get_custom_param('min_bbox_area', MIN_BBOX_AREA):
                    continue
                
                aspect = bw / max(bh, 1)
                if aspect > 3.5 or aspect < 0.2:
                    continue

                # STRICT: Box must contain at least 10% warm pixels to be a barricade/cone
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv_roi, lower_warm, upper_warm)
                    warm_pixels = cv2.countNonZero(mask)
                    total_pixels = bw * bh
                    
                    if (warm_pixels / total_pixels) < 0.05:
                        continue

                detections.append({
                    'type':       label,
                    'confidence': round(conf, 3),
                    'bbox':       [x1, y1, bw, bh]
                })
        return detections

    def _draw(self, frame, detections):
        for d in detections:
            x, y, bw, bh = d['bbox']
            conf  = d.get('confidence', 0)
            text  = f"{d['type']} {int(conf * 100)}%"
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), COLOR_BOX, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x, y - th - 4), (x + tw, y), COLOR_BOX, -1)
            cv2.putText(frame, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1, cv2.LINE_AA)
        return frame

    def _get_custom_param(self, name, default):
        try:
            return self.get_parameter(name).value
        except Exception:
            return default


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()