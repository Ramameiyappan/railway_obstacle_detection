import cv2
import numpy as np

def testing(image, track_model, obstacle_model):
    test_image = image.copy()

    #track detetcion
    track_results = track_model.predict(source=test_image)
    track_res = track_results[0]

    #obstacle detection
    obstacle_results = obstacle_model.predict(source=test_image)
    boxes = obstacle_results[0].boxes

    #creating maks same size as that of input to oocupy the area of track and obstacle
    mask = np.zeros(test_image.shape[:2], dtype=np.uint8)
    h, w = mask.shape

    #filling the mask with track class
    if track_res.masks is not None and track_res.boxes is not None:
        classes = track_res.boxes.cls.cpu().numpy().astype(int)
        names = track_res.names

        for seg, cls_id in zip(track_res.masks.xy, classes):
            label = names[int(cls_id)]
            if label != "track":
                continue
            poly = np.array(seg, dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)

    #getting obstacle which are greater than 35 as confidence
    if boxes is not None:
        conf = boxes.conf.cpu().numpy()
        keep = conf >= 0.35
        obstacle_boxes = boxes.xyxy.cpu().numpy().astype(int)[keep]
        obstacle_classes = boxes.cls.cpu().numpy().astype(int)[keep]
    else:
        obstacle_boxes = []
        obstacle_classes = []

    obstacle_names = obstacle_results[0].names
    obstacle_on_track = False
    obstacle_types_on_track = []

    #checking obstacle in the traack by using mask
    for i, box in enumerate(obstacle_boxes):
        x1, y1, x2, y2 = box

        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        obstacle_region = mask[y1:y2, x1:x2]
        if obstacle_region.size == 0:
            continue

        overlap_pixels = np.count_nonzero(obstacle_region)
        overlap_ratio = overlap_pixels / obstacle_region.size

        if overlap_ratio >= 0.10:
            obstacle_on_track = True
            class_id = obstacle_classes[i]
            obstacle_types_on_track.append(
                obstacle_names[int(class_id)]
            )

    #forming message to be sent
    if obstacle_on_track:
        unique_obstacles = list(set(obstacle_types_on_track))
        message = f"{', '.join(unique_obstacles)} obstacle detected on track"
        status = "danger"
    else:
        message = "No obstacle on track"
        status = "safe"

    #making image with label from the model
    combined = obstacle_results[0].plot(img=test_image.copy())
    track_mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(combined, 0.8, track_mask_colored, 0.4, 0)

    return overlay, message, status
