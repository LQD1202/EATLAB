from ultralytics import YOLO
import subprocess
import cv2
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
import argparse
def model_index(index):
    if index == 1:
        # === BƯỚC 1: CẮT VIDEO ===
        input_video = "./dataset/1461_CH01_20250607193711_203711.mp4"
        output_video_15min = "./dataset/temp_15min.mp4"
        start_time = "00:10:00"
        duration = "00:05:00"

        print("🔪 Cắt video bằng ffmpeg...")
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", start_time,
            "-i", input_video,
            "-t", duration,
            "-c", "copy",
            output_video_15min
        ], check=True)
        print("✅ Cắt video thành công:", output_video_15min)

        # === BƯỚC 2: KHỞI TẠO MODEL VÀ TRACKER ===
        model_path = "./yolov12m-cam1/yolov12m-cam13/weights/best.pt"
        model = YOLO(model_path)
        tracker = DeepSort(max_age=300)

        # === BƯỚC 3: SETUP VIDEO OUTPUT ===
        cap = cv2.VideoCapture(output_video_15min)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_video = "./dataset/temp_15min_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # === VÙNG QUAN TÂM ===
        x_center = (width // 2) + 100
        y_center = (height // 2) + 200
        y_thresh = 200

        # === BIẾN THEO DÕI ===
        frame_count = 0
        count_pizza = 0
        tracked_ids = set()

        # === THAM SỐ MODEL ===
        class_id_pizza = 67  # ID class "pizza"
        conf_thresh = 0.9

        # === VÒNG LẶP DỰ ĐOÁN ===
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(
                frame,
                conf=conf_thresh,
                classes=[class_id_pizza],
                agnostic_nms=False,
                max_det=100,
                device="cuda:1",
                verbose=False
            )

            detections = []
            boxes = results[0].boxes
            annotated_frame = frame.copy()

            if boxes is not None and boxes.xyxy is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].item())
                    cls = int(box.cls[0].item())
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], conf, str(cls)))

            # === CẬP NHẬT TRACKING ===
            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed() or track.track_id is None:
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())

                # Tìm độ tin cậy gần đúng
                conf = 0.0
                for det in detections:
                    bbox, det_conf, det_cls = det
                    x, y, w, h = bbox
                    if abs(x - x1) < 10 and abs(y - y1) < 10:
                        conf = det_conf
                        break

                # Tính tâm object
                x_center_obj = (x1 + x2) // 2
                y_center_obj = (y1 + y2) // 2

                # Lọc object bên phải khu vực quan tâm
                if x_center_obj > x_center and y_thresh < y_center_obj < y_center and conf >= conf_thresh:
                    if track_id not in tracked_ids:
                        tracked_ids.add(track_id)
                        count_pizza += 1

                    # Vẽ box + ID
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(annotated_frame, f"ID {track_id} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Hiển thị số lượng pizza đã đếm
            cv2.putText(annotated_frame, f"Pizza Count: {count_pizza}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            out.write(annotated_frame)
            frame_count += 1
            print(f"🧠 Đã xử lý frame {frame_count}")

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"🎬 Hoàn tất! Video lưu tại: {output_video}")
    elif index == 2:
        # === BƯỚC 1: CẮT VIDEO ===
        input_video = "./dataset/1465_CH02_20250607170555_172408.mp4"
        output_video_15min = "./dataset/temp_15min.mp4"
        start_time = "00:04:50"
        duration = "00:01:00"

        print("🔪 Cắt video bằng ffmpeg...")
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", start_time,
            "-i", input_video,
            "-t", duration,
            "-c", "copy",
            output_video_15min
        ], check=True)
        print("✅ Cắt video thành công:", output_video_15min)

        # === BƯỚC 2: KHỞI TẠO MODEL VÀ TRACKER ===
        model_path = "./yolov12m-cam2/yolov12m-cam26/weights/best.pt"
        model = YOLO(model_path)
        tracker = DeepSort(max_age=300)

        # === BƯỚC 3: ĐỌC VIDEO VÀ TRACK ===
        cap = cv2.VideoCapture(output_video_15min)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_video = "./dataset/temp_15min_detected_cam2.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # === THƯ MỤC LƯU FRAME ===
        output_frame_dir = "./dataset/test/frames"
        os.makedirs(output_frame_dir, exist_ok=True)

        frame_count = 0
        tracked_ids = set()

        # === ĐƯỜNG CHÉO: y = -x + b ===
        b = height + 100  # điều chỉnh nếu muốn đường chéo cao hơn hoặc thấp hơn

        # === THAM SỐ PHÁT HIỆN ===
        conf_thresh = 0.9
        target_cls = 0  # lớp cần phát hiện (ví dụ: 0 = pizza)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(
                frame,
                conf=conf_thresh,
                classes=[target_cls],
                agnostic_nms=False,
                max_det=100,
                device="cuda:1",
                verbose=False
            )

            detections = []
            boxes = results[0].boxes

            if boxes is not None and boxes.xyxy is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].item())
                    cls = int(box.cls[0].item())
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], conf, str(cls)))

            tracks = tracker.update_tracks(detections, frame=frame)
            annotated_frame = frame.copy()

            pt1 = (0, b)
            pt2 = (width, -width + b)
            cv2.line(annotated_frame, pt1, pt2, (255, 0, 255), 2)

            for track in tracks:
                if not track.is_confirmed() or track.track_id is None:
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())

                # Tìm lại độ tin cậy gần đúng từ detection
                conf = 0.0
                for det in detections:
                    bbox, det_conf, det_cls = det
                    x, y, w, h = bbox
                    if abs(x - x1) < 10 and abs(y - y1) < 10:
                        conf = det_conf
                        break

                # Tâm object
                x_center_obj = (x1 + x2) // 2
                y_center_obj = (y1 + y2) // 2

                # Kiểm tra nếu object nằm bên phải (dưới) đường chéo
                if conf >= conf_thresh:
                    tracked_ids.add(track_id)

                    # Vẽ bbox + ID
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(annotated_frame, f"ID {track_id} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)

            # Vẽ tổng số object đã đếm
            cv2.putText(annotated_frame, f"Pizza Count: {len(tracked_ids)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # Ghi video và lưu frame
            out.write(annotated_frame)
            frame_filename = os.path.join(output_frame_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, annotated_frame)

            frame_count += 1

        # === KẾT THÚC ===
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"🎬 Done! Output video saved to: {output_video}")
        print(f"🖼️ Các frame đã được lưu tại: {output_frame_dir}")
        print(f"📦 Tổng số object duy nhất nằm bên phải đường chéo: {len(tracked_ids)}")
    elif index == 3:
        # === BƯỚC 1: CẮT VIDEO ===
        input_video = "./dataset/1462_CH04_20250607210159_211703.mp4"
        output_video_15min = "./dataset/temp_15min.mp4"
        start_time = "00:10:00"
        duration = "00:01:00"

        print("🔪 Cắt video bằng ffmpeg...")
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", start_time,
            "-i", input_video,
            "-t", duration,
            "-c", "copy",
            output_video_15min
        ], check=True)
        print("✅ Cắt video thành công:", output_video_15min)

        # === BƯỚC 2: KHỞI TẠO MODEL VÀ TRACKER ===
        model_path = "./yolov12m-cam3/yolov12m-cam3/weights/best.pt"
        model = YOLO(model_path)
        tracker = DeepSort(max_age=30)

        # === THƯ MỤC LƯU FRAME ===
        output_frame_dir = "./dataset/frames"
        os.makedirs(output_frame_dir, exist_ok=True)

        # === VIDEO GHI OUTPUT ===
        cap = cv2.VideoCapture(output_video_15min)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = "./dataset/temp_15min_detected_cam3.mp4"
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        frame_count = 0
        tracked_ids = set()

        CONF_THRESHOLD = 0.90  # Chỉ vẽ nếu conf > 0.90

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(
                frame,
                conf=CONF_THRESHOLD,
                classes=[0],
                agnostic_nms=False,
                max_det=100,
                device="cuda:1",
                verbose=False,
                iou=0.1
            )

            detections = []
            conf_map = {}  # lưu box: conf để vẽ về sau

            boxes = results[0].boxes
            if boxes is not None and boxes.xyxy is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].item())
                    cls = int(box.cls[0].item())
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], conf, str(cls)))
                    conf_map[(int(x1), int(y1), int(x2), int(y2))] = conf

            # === TRACKING ===
            tracks = tracker.update_tracks(detections, frame=frame)
            annotated_frame = frame.copy()

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                track_box = (x1, y1, x2, y2)

                # Tìm lại conf gần đúng theo IOU hoặc vị trí box
                matched_conf = 0.0
                for (bx1, by1, bx2, by2), c in conf_map.items():
                    if abs(bx1 - x1) < 10 and abs(by1 - y1) < 10:
                        matched_conf = c
                        break

                if matched_conf >= CONF_THRESHOLD:
                    tracked_ids.add(track_id)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(annotated_frame, f"ID {track_id} ({matched_conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)

            # Hiển thị đếm số lượng object duy nhất
            cv2.putText(annotated_frame, f"Pizza Count: {len(tracked_ids)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # Ghi video và lưu frame
            out.write(annotated_frame)
            frame_path = os.path.join(output_frame_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_path, annotated_frame)

            frame_count += 1

        # === KẾT THÚC ===
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"🎬 Done! Output video saved to: {output_video}")
        print(f"🖼️ Các frame đã được lưu tại: {output_frame_dir}")
    elif index == 4:

        # === BƯỚC 1: CẮT VIDEO ===
        input_video = "./dataset/1464_CH02_20250607180000_190000.mp4"
        output_video_15min = "./temp_15min.mp4"
        start_time = "00:22:30"
        duration = "00:01:00"

        print("🔪 Cắt video bằng ffmpeg...")
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", start_time,
            "-i", input_video,
            "-t", duration,
            "-c", "copy",
            output_video_15min
        ], check=True)
        print("✅ Cắt video thành công:", output_video_15min)

        # === BƯỚC 2: KHỞI TẠO MODEL VÀ TRACKER ===
        model_path = "./yolov12m-cam4/yolov12m-cam46/weights/best.pt"
        model = YOLO(model_path)
        tracker = DeepSort(max_age=30)

        # === THƯ MỤC LƯU FRAME ===
        output_frame_dir = "./frames"
        os.makedirs(output_frame_dir, exist_ok=True)

        # === VIDEO GHI OUTPUT ===
        cap = cv2.VideoCapture(output_video_15min)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = "./dataset/temp_15min_detected_cam4.mp4"
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        frame_count = 0
        tracked_ids = set()

        CONF_THRESHOLD = 0.90  # Chỉ vẽ nếu conf > 0.90

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(
                frame,
                conf=CONF_THRESHOLD,
                classes=[0],
                agnostic_nms=False,
                max_det=100,
                device="cuda:1",
                verbose=False,
                iou=0.1
            )

            detections = []
            conf_map = {}  # lưu box: conf để vẽ về sau

            boxes = results[0].boxes
            if boxes is not None and boxes.xyxy is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].item())
                    cls = int(box.cls[0].item())
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], conf, str(cls)))
                    conf_map[(int(x1), int(y1), int(x2), int(y2))] = conf

            # === TRACKING ===
            tracks = tracker.update_tracks(detections, frame=frame)
            annotated_frame = frame.copy()

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                track_box = (x1, y1, x2, y2)

                # Tìm lại conf gần đúng theo IOU hoặc vị trí box
                matched_conf = 0.0
                for (bx1, by1, bx2, by2), c in conf_map.items():
                    if abs(bx1 - x1) < 10 and abs(by1 - y1) < 10:
                        matched_conf = c
                        break

                if matched_conf >= CONF_THRESHOLD:
                    tracked_ids.add(track_id)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(annotated_frame, f"ID {track_id} ({matched_conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)

            # Hiển thị đếm số lượng object duy nhất
            cv2.putText(annotated_frame, f"Pizza Count: {len(tracked_ids)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # Ghi video và lưu frame
            out.write(annotated_frame)
            frame_path = os.path.join(output_frame_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_path, annotated_frame)

            frame_count += 1

        # === KẾT THÚC ===
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"🎬 Done! Output video saved to: {output_video}")
        print(f"🖼️ Các frame đã được lưu tại: {output_frame_dir}")
    elif index == 5:

        # === BƯỚC 1: CẮT VIDEO ===
        input_video = "./dataset/1465_CH02_20250607170555_172408.mp4"
        output_video_15min = "./dataset/temp_15min.mp4"
        start_time = "00:04:50"
        duration = "00:01:00"

        print("🔪 Cắt video bằng ffmpeg...")
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", start_time,
            "-i", input_video,
            "-t", duration,
            "-c", "copy",
            output_video_15min
        ], check=True)
        print("✅ Cắt video thành công:", output_video_15min)

        # === BƯỚC 2: KHỞI TẠO MODEL VÀ TRACKER ===
        model_path = "./yolov12m-cam5/yolov12m-cam56/weights/best.pt"
        model = YOLO(model_path)
        tracker = DeepSort(max_age=300)

        # === BƯỚC 3: ĐỌC VIDEO VÀ TRACK ===
        cap = cv2.VideoCapture(output_video_15min)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_video = "./dataset/temp_15min_detected_cam5.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # === THƯ MỤC LƯU FRAME ===
        output_frame_dir = "./dataset/test/frames"
        os.makedirs(output_frame_dir, exist_ok=True)

        frame_count = 0
        tracked_ids = set()

        # === THAM SỐ GIỚI HẠN KHU VỰC ===
        y_center = height // 2                   # chỉ lấy object phía dưới
        x_line = width // 2 + 200                # chỉ lấy object bên phải đường thẳng này

        # === THAM SỐ PHÁT HIỆN ===
        conf_thresh = 0.9
        target_cls = 0  # lớp cần phát hiện

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(
                frame,
                conf=conf_thresh,
                classes=[target_cls],
                agnostic_nms=False,
                max_det=100,
                device="cuda:1",
                verbose=False
            )

            detections = []
            boxes = results[0].boxes

            if boxes is not None and boxes.xyxy is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].item())
                    cls = int(box.cls[0].item())
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], conf, str(cls)))

            tracks = tracker.update_tracks(detections, frame=frame)
            annotated_frame = frame.copy()

            # === Vẽ vùng lọc: đường ngang + đường dọc
            cv2.line(annotated_frame, (0, y_center), (width, y_center), (255, 0, 0), 2)  # ngang (xanh dương)
            cv2.line(annotated_frame, (x_line, 0), (x_line, height), (255, 255, 0), 2)   # dọc (cyan)

            for track in tracks:
                if not track.is_confirmed() or track.track_id is None:
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())

                # Tìm lại độ tin cậy gần đúng từ detection
                conf = 0.0
                for det in detections:
                    bbox, det_conf, det_cls = det
                    x, y, w, h = bbox
                    if abs(x - x1) < 10 and abs(y - y1) < 10:
                        conf = det_conf
                        break

                # Tâm đối tượng
                x_center_obj = (x1 + x2) // 2
                y_center_obj = (y1 + y2) // 2

                # Chỉ đếm nếu object nằm bên phải x_line và bên dưới y_center
                if conf >= conf_thresh and y_center_obj > y_center and x_center_obj > x_line:
                    tracked_ids.add(track_id)

                    # Vẽ bbox và ID
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(annotated_frame, f"ID {track_id} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)

            # Ghi tổng số ID duy nhất
            cv2.putText(annotated_frame, f"Pizza Count: {len(tracked_ids)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # Ghi video và lưu frame
            out.write(annotated_frame)
            frame_filename = os.path.join(output_frame_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, annotated_frame)

            frame_count += 1

        # === KẾT THÚC ===
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"🎬 Done! Output video saved to: {output_video}")
        print(f"🖼️ Các frame đã được lưu tại: {output_frame_dir}")
        print(f"📦 Tổng số object duy nhất được track (bên phải đường chia): {len(tracked_ids)}")
    else:
        def is_point_below_line(x, y, pt1, pt2):
            x1, y1 = pt1
            x2, y2 = pt2

            if x1 == x2:
                # Đường thẳng đứng => không xác định "dưới"
                return x < x1  # hoặc x > x1 tùy logic bạn cần
            else:
                # Tính y trên đường tại hoành độ x
                y_on_line = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
                return y > y_on_line  # trong ảnh: trục y hướng xuống => "dưới" là y lớn hơn

        # === BƯỚC 1: CẮT VIDEO ===
        input_video = "./dataset/1467_CH04_20250607180000_190000.mp4"
        output_video_15min = "./dataset/temp_15min.mp4"
        start_time = "00:01:10"
        duration = "00:01:00"

        print("🔪 Cắt video bằng ffmpeg...")
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", start_time,
            "-i", input_video,
            "-t", duration,
            "-c", "copy",
            output_video_15min
        ], check=True)
        print("✅ Cắt video thành công:", output_video_15min)

        # === BƯỚC 2: KHỞI TẠO MODEL VÀ TRACKER ===
        model_path = "./yolov12m-cam6/yolov12m-cam64/weights/best.pt"
        model = YOLO(model_path)
        tracker = DeepSort(max_age=300)

        # === BƯỚC 3: ĐỌC VIDEO VÀ TRACK ===
        cap = cv2.VideoCapture(output_video_15min)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_video = "./dataset/temp_15min_detected_cam6.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # === THƯ MỤC LƯU FRAME ===
        output_frame_dir = "./dataset/test/frames"
        os.makedirs(output_frame_dir, exist_ok=True)

        frame_count = 0
        tracked_ids = set()

        # === ĐƯỜNG CHÉO: y = -x + b ===
        b = height + 300  # điều chỉnh nếu muốn đường chéo cao hơn hoặc thấp hơn

        # === THAM SỐ PHÁT HIỆN ===
        conf_thresh = 0.9
        target_cls = 0  # lớp cần phát hiện (ví dụ: 0 = pizza)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(
                frame,
                conf=conf_thresh,
                classes=[target_cls],
                agnostic_nms=False,
                max_det=100,
                device="cuda:1",
                verbose=False
            )

            detections = []
            boxes = results[0].boxes

            if boxes is not None and boxes.xyxy is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].item())
                    cls = int(box.cls[0].item())
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], conf, str(cls)))

            tracks = tracker.update_tracks(detections, frame=frame)
            annotated_frame = frame.copy()

            pt1 = (0, b)
            pt2 = (width, max(0, -width + b))  # đảm bảo toạ độ không âm

            # Vẽ đường
            cv2.line(annotated_frame, pt1, pt2, (255, 0, 255), 2)

            for track in tracks:
                if not track.is_confirmed() or track.track_id is None:
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())

                # Tìm lại độ tin cậy gần đúng từ detection
                conf = 0.0
                for det in detections:
                    bbox, det_conf, det_cls = det
                    x, y, w, h = bbox
                    if abs(x - x1) < 10 and abs(y - y1) < 10:
                        conf = det_conf
                        break

                # Tâm object
                x_center_obj = (x1 + x2) // 2
                y_center_obj = (y1 + y2) // 2

                # Kiểm tra nếu object nằm bên phải (dưới) đường chéo
                if conf >= conf_thresh and is_point_below_line(x, y, pt1,pt2):
                    tracked_ids.add(track_id)

                    # Vẽ bbox + ID
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(annotated_frame, f"ID {track_id} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)

            # Vẽ tổng số object đã đếm
            cv2.putText(annotated_frame, f"Pizza Count: {len(tracked_ids)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # Ghi video và lưu frame
            out.write(annotated_frame)
            frame_filename = os.path.join(output_frame_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, annotated_frame)

            frame_count += 1

        # === KẾT THÚC ===
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"🎬 Done! Output video saved to: {output_video}")
        print(f"🖼️ Các frame đã được lưu tại: {output_frame_dir}")
        print(f"📦 Tổng số object duy nhất nằm bên phải đường chéo: {len(tracked_ids)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chọn video để inference")
    parser.add_argument(
        "--video", 
        type=int, 
        choices=range(1, 7), 
        required=True,
        help="Chọn video từ 1 đến 6",
        default=1
    )
    args = parser.parse_args()

    video_path = model_index(args.video)

    if video_path is None:
        print("Video không tồn tại!")
    else:
        print(f"Đang chạy inference trên: {video_path}")