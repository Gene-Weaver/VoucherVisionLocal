"""TextCollage: crop the labels (and barcodes, maps) out of a specimen sheet and pack them
into one compact image for OCR.

VVL: the boxes come from LeafMachine3's archival detector (textcollage.detector,
Core ML on macOS, onnxruntime elsewhere). The collage itself is TextCollage's.
"""
import base64
import json
import threading

import cv2
import numpy as np

from textcollage.detector import imread


def _class_key(name):
    """The model's class name -> TextCollage's: "Attached Item" -> "attached_item"."""
    return str(name).strip().lower().replace(" ", "_")


def _encode(image, output_path):
    """{"position_original", "position_collage", "base64image_text_collage"} for an image
    used as-is; written to `output_path` instead of base64 when one is given."""
    output = {"position_original": {}, "position_collage": {}, "base64image_text_collage": None}
    success, jpg_array = cv2.imencode('.jpg', image)
    if not success: raise RuntimeError("Failed to encode collage image to JPG.")
    jpg_bytes = jpg_array.tobytes()
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(jpg_bytes)
    else:
        output['base64image_text_collage'] = base64.b64encode(jpg_bytes).decode('utf-8')
    return output


def whole_image(image_path, output_path=None):
    """The result shape of CollageEngine.run, for the whole image (no detector needed)."""
    original_image = imread(image_path)
    if original_image is None: raise ValueError(f"Image not found at {image_path}")
    return _encode(original_image, output_path)


class CollageEngine:
    # Every class the detector knows, in TextCollage's spelling.
    all_possible_classes = ['ruler', 'barcode', 'colorcard', 'label', 'map', 'envelope', 'photo', 'attached_item', 'weights']

    def __init__(
        self,
        detector,
        collage_classes,
        engine="gemini",
        hide_long_objects=False,
        draw_overlay=False,
        output_path="collage_output.jpg", # Can be None
    ):
        """`detector` is a textcollage.detector.ArchivalDetector (or anything with
        detect(image_bgr) -> [Detection(cls, score, box)])."""
        self.engine = engine
        self.hide_long_objects = hide_long_objects
        self.draw_overlay = draw_overlay
        self.output_path = output_path
        self.collage_classes = collage_classes
        self.detector = detector
        self.inference_lock = threading.Lock()

    def get_bounding_boxes(self, image_path):
        original_image = imread(image_path)
        if original_image is None: raise ValueError(f"Image not found at {image_path}")
        with self.inference_lock:
            detections = self.detector.detect(original_image)
        final_boxes = {name: [] for name in self.all_possible_classes}
        for det in detections:
            key = _class_key(det.cls)
            if key not in final_boxes:
                continue
            x1, y1, x2, y2 = (int(v) for v in det.box)
            if x2 > x1 and y2 > y1:
                final_boxes[key].append([x1, y1, x2, y2])
        return original_image, final_boxes

    def merge_overlapping_boxes(self, boxes):
        if not boxes: return []
        box_list = [list(b) for b in boxes]
        while True:
            merged_in_pass = False; i = 0
            while i < len(box_list):
                j = i + 1
                while j < len(box_list):
                    box1, box2 = box_list[i], box_list[j]
                    if (box1[0] < box2[2] and box1[2] > box2[0] and box1[1] < box2[3] and box1[3] > box2[1]):
                        box_list[i] = [min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])]
                        box_list.pop(j); merged_in_pass = True; break
                    else: j += 1
                if merged_in_pass: break
                else: i += 1
            if not merged_in_pass: break
        return box_list

    def partition_by_aspect_ratio(self, boxes_by_class, threshold=2.0):
        normal_boxes = {n: [] for n in self.all_possible_classes}; long_boxes = {n: [] for n in self.all_possible_classes}
        for class_name, boxes in boxes_by_class.items():
            for box in boxes:
                w, h = box[2] - box[0], box[3] - box[1]; ratio = w / h if h > 0 else float("inf")
                if ratio > threshold: long_boxes[class_name].append(box)
                else: normal_boxes[class_name].append(box)
        return normal_boxes, long_boxes

    def _create_condensed_collage_from_crops(self, crops):
        if not crops: return None, {name: [] for name in self.all_possible_classes}
        positions = {name: [] for name in self.all_possible_classes}; rows = []
        crops.sort(key=lambda c: c["box"][1])
        for crop in crops:
            placed = False; y1, y2 = crop["box"][1], crop["box"][3]
            for row in rows:
                if max(row["y_min"], y1) < min(row["y_max"], y2):
                    row["crops"].append(crop); row["y_min"] = min(row["y_min"], y1); row["y_max"] = max(row["y_max"], y2); placed = True; break
            if not placed: rows.append({"crops": [crop], "y_min": y1, "y_max": y2})

        row_dims, max_w, total_h = [], 0, 0
        for row in rows:
            row["crops"].sort(key=lambda c: c["box"][0])
            rh = max(c["img"].shape[0] for c in row["crops"]); rw = sum(c["img"].shape[1] for c in row["crops"])
            row_dims.append((rw, rh)); max_w = max(max_w, rw); total_h += rh
            
        canvas = np.zeros((total_h, max_w, 3), dtype=np.uint8)
        y_off = 0
        for i, row in enumerate(rows):
            x_off = 0
            for c in row["crops"]:
                h, w = c["img"].shape[:2]; canvas[y_off:y_off + h, x_off:x_off + w] = c["img"]
                positions[c["class"]].append([x_off, y_off, x_off + w, y_off + h])
                x_off += w
            y_off += row_dims[i][1]
        return canvas, positions

    def _append_long_objects(self, base_collage, base_positions, long_crops):
        if not long_crops: return base_collage, base_positions
        long_strip, long_pos_relative = self._create_condensed_collage_from_crops(long_crops)
        base_h, base_w = (base_collage.shape[:2] if base_collage is not None else (0,0))
        strip_h, strip_w = long_strip.shape[:2]
        final_w = max(base_w, strip_w); final_h = base_h + strip_h
        final_canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)

        if base_collage is not None: final_canvas[0:base_h, 0:base_w] = base_collage
        final_canvas[base_h:final_h, 0:strip_w] = long_strip
        
        final_positions = base_positions
        for class_name, bboxes in long_pos_relative.items():
            if bboxes:
                for box in bboxes:
                    final_positions[class_name].append([box[0], box[1] + base_h, box[2], box[3] + base_h])
        return final_canvas, final_positions

    def resize_for_engine(self, image):
        h, w = image.shape[:2]; scale = 1.0
        if self.engine == "gemini": return image, scale
        elif self.engine == "claude":
            longest = max(h, w)
            if longest > 1568: scale = 1568 / longest; image = cv2.resize(image, (int(w * scale), int(h * scale)))
        elif self.engine == "gpt":
            longest = max(h, w)
            if longest > 2048: scale = 2048 / longest; image = cv2.resize(image, (int(w * scale), int(h * scale))); h, w = image.shape[:2]
            shortest = min(h, w)
            if shortest > 768: step2 = 768 / shortest; scale *= step2; image = cv2.resize(image, (int(w * step2), int(h * step2)))
        return image, scale

    def run(self, image_path):
        original_image, raw_boxes = self.get_bounding_boxes(image_path)
        merged_boxes = {c: self.merge_overlapping_boxes(b) for c, b in raw_boxes.items() if b}
        
        normal_boxes, long_boxes = self.partition_by_aspect_ratio(merged_boxes) if self.hide_long_objects else (merged_boxes, {n: [] for n in self.all_possible_classes})
        final_output = {"position_original": merged_boxes, "position_collage": {}, "base64image_text_collage": None}

        crops_for_collage = []
        for class_name in self.collage_classes:
            if class_name in normal_boxes:
                for box in normal_boxes[class_name]:
                    x1, y1, x2, y2 = map(int, box); crops_for_collage.append({"img": original_image[y1:y2, x1:x2], "box": box, "class": class_name})
            if self.hide_long_objects and class_name in long_boxes:
                for box in long_boxes[class_name]:
                    x1, y1, x2, y2 = map(int, box); crops_for_collage.append({"img": original_image[y1:y2, x1:x2], "box": box, "class": class_name})

        collage, positions = self._create_condensed_collage_from_crops(crops_for_collage)

        if collage is None: raise RuntimeError("No collage could be created.")
        
        collage, scale = self.resize_for_engine(collage)
        final_output["position_collage"] = {c: [[int(coord * scale) for coord in box] for box in bboxes] for c, bboxes in positions.items()}

        if self.draw_overlay: collage = self.draw_overlay_on_collage(collage, final_output["position_collage"])
        
        # Always encode the image to a JPG byte stream in memory
        success, jpg_array = cv2.imencode('.jpg', collage)
        if not success: raise RuntimeError("Failed to encode collage image to JPG.")
        
        # Convert numpy array to actual bytes
        jpg_bytes = jpg_array.tobytes()
        
        if self.output_path:
            # If a path is provided, write the bytes to disk
            with open(self.output_path, 'wb') as f:
                f.write(jpg_bytes)
        else:
            # If no path, encode bytes to base64 and add to JSON output
            final_output['base64image_text_collage'] = base64.b64encode(jpg_bytes).decode('utf-8')

        return final_output
    
    def run_fake(self, image_path): # Produce the same output without making the collage
        return whole_image(image_path, self.output_path)

    def draw_overlay_on_collage(self, image, positions):
        overlay = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for class_name, bboxes in positions.items():
            if not bboxes: continue
            for box in bboxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {x1},{y1},{x2},{y2}"
                (w, h), _ = cv2.getTextSize(label, font, 0.5, 1)
                cv2.rectangle(overlay, (x1, y1 - h - 4), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(overlay, label, (x1, y1 - 2), font, 0.5, (0, 0, 0), 1)
        return overlay
