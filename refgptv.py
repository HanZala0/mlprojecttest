import streamlit as st
import cv2
from ultralytics import YOLO
import supervision as sv
import tempfile
import os
import numpy as np

st.title("YOLOv8 Object Detection, Tracking & Counting")
st.markdown("Upload a video to detect, track, and count objects using YOLOv8 and Supervision.")

# Sidebar options
with st.sidebar:
    st.header("Configuration")
    model_type = st.selectbox(
        "Select YOLOv8 model",
        ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt")
    )
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    selected_classes = st.multiselect(
        "Select classes to detect",
        options=list(YOLO(model_type).model.names.values()),
        default=["car", "person"]
    )

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()

    # Initialize YOLO model
    model = YOLO(model_type)

    # Get class IDs for selected classes
    if selected_classes:
        class_ids = [
            list(model.model.names.keys())[list(model.model.names.values()).index(cls)] 
            for cls in selected_classes
        ]
    else:
        class_ids = None

    # Initialize ByteTrack tracker
    tracker = sv.ByteTrack()

    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)
    trace_annotator = sv.TraceAnnotator(thickness=4)

    # Frame generator
    video_info = sv.VideoInfo.from_video_path(video_path)
    generator = sv.get_video_frames_generator(video_path)

    # Line counter setup (horizontal line in the middle)
    #line_start = sv.Point(0, int(video_info.height / 2))
    #line_end = sv.Point(video_info.width, int(video_info.height / 2))
    line_start = sv.Point(0, 1500)
    line_end = sv.Point(3840, 1500)
    line_counter = sv.LineZone(start=line_start, end=line_end)
    line_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

    # Counting dictionary initialization
    counting_dict = {cls: 0 for cls in selected_classes} if selected_classes else {}

    # Streamlit placeholders
    counting_placeholder = st.empty()
    stframe = st.empty()

    for frame in generator:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter detections by confidence and class
        detections = detections[detections.confidence > confidence_threshold]
        if class_ids:
            detections = detections[np.isin(detections.class_id, class_ids)]

        # Update tracker with detections
        detections = tracker.update_with_detections(detections)

        # Check line crossing and update counts
        triggered_ids = line_counter.trigger(detections)

        for tid in triggered_ids:
            # Find detection corresponding to triggered tracker id
            idx = np.where(detections.tracker_id == tid)[0]
            if len(idx) > 0:
                class_id = detections.class_id[idx[0]]
                class_name = model.model.names[class_id]
                # Update count only if class is in selected classes
                if class_name in counting_dict:
                    counting_dict[class_name] += 1

        # Prepare labels for each detection
        labels = [
            f"ID {tracker_id} {model.model.names[class_id]} {confidence:.2f}"
            for tracker_id, class_id, confidence in
            zip(detections.tracker_id, detections.class_id, detections.confidence)
        ]

        # Annotate frame
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = bounding_box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
        annotated_frame = line_annotator.annotate(annotated_frame, line_counter)

        # Show annotated frame in Streamlit
        stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        # Display counts
        counting_text = "\n".join([f"{cls}: {count}" for cls, count in counting_dict.items()])
        counting_placeholder.text(f"Object Counts (crossing the line):\n{counting_text}")

    # Clean up temporary file
    os.unlink(video_path)
