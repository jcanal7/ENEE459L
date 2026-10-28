import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection

#Non-Maximum Suppression
# box scores: x1,y1,x2,y2,score
# topk=-1, -1 means no limit
# IoU= intersection area/union area
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):

    scores = box_scores[:, -1] #last column
    boxes = box_scores[:, :-1] # up to last column

    indexes = np.argsort(scores) #sort by confidence in ascending order
    indexes = indexes[-candidate_size:] #limit candidates

    picked = []

    while len(indexes) > 0:
        current = indexes[-1] #get box with highest score
        picked.append(current) #save it
        indexes = indexes[:-1] #remove index from candidates

        if top_k > 0 and len(picked) >= top_k:
            break

        if len(indexes) == 0:
            break

        current_box = boxes[current]
        remaining_boxes = boxes[indexes]

        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

        intersection_w = np.maximum(0, x2 - x1)
        intersection_h = np.maximum(0, y2 - y1)

        intersection = intersection_w * intersection_h

        current_area = (
            (current_box[2] - current_box[0]) *
            (current_box[3] - current_box[1])
        )

        remaining_areas = (
            (remaining_boxes[:, 2] - remaining_boxes[:, 0]) *
            (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        )

        union = current_area + remaining_areas - intersection

        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0
        )

        indexes = indexes[iou <= iou_threshold] #boolean indexing, keep trues, discard falses

    return box_scores[picked, :]

def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):

    # Step 1: Load image and save original dimensions
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Step 2: Resize and normalize
    resized = image.resize((320, 240))
    img_array = np.asarray(resized, dtype=np.float32)
    img_array = (img_array - 127.0) / 128.0

    # Step 3: Convert HWC to NCHW
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    # Step 4: Run ONNX inference
    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=available_providers
    )

    input_name = session.get_inputs()[0].name

    confidences, boxes = session.run(
        None,
        {input_name: img_array}
    )

    # Step 5: Remove batch dimension and filter scores
    confidences = confidences[0]
    boxes = boxes[0]

    face_scores = confidences[:, 1]

    mask = face_scores >= conf_threshold

    filtered_boxes = boxes[mask]
    filtered_scores = face_scores[mask]

    box_scores = np.column_stack(
        (filtered_boxes, filtered_scores)
    )

    # Step 6: Remove overlapping detections
    picked = hard_nms(box_scores, iou_threshold=0.3)

    # Step 7: Draw detections on the original image
    draw = ImageDraw.Draw(image)

    for detection in picked:
        box = detection[:4]
        score = detection[4]

        box = box * np.array([width, height, width, height])

        x1, y1, x2, y2 = box.astype(int)

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="lime",
            width=3
        )

        draw.text(
            (x1, max(0, y1-13)),
            f"Face: {score:.2f}",
            fill="lime"
        )

    # Step 8: Save the annotated image
    image.save(output_path)

# Edge detection
def non_maximum_suppression(magnitude, angle):
    #magnitude - how strong each edge is at each pixel
    #angle - gradient direction at each pixel, in radians

    # Step 1: Initialize output array, 3x3 array
    height, width = magnitude.shape
    output = np.zeros((height, width), dtype=np.float32)

    # Step 2: Convert angles to degrees
    angle = np.rad2deg(angle) % 180

    # Step 3: Loop through internal pixels
    for i in range(1, height - 1):
        for j in range(1, width - 1):

            # Step 4: Find neighboring pixels based on angle
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] < 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]

            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]

            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i - 1, j]
                r = magnitude[i + 1, j]

            else:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            # Step 5: Keep only local maximums
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                output[i, j] = magnitude[i, j]

    # Step 6: Return thinned magnitude array (making image edges thinner)
    return output

def hysteresis_thresholding(img, low_thresh, high_thresh):

    # Step 1: Define pixel values
    strong = 255
    weak = 50
    suppressed = 0

    # Step 2: Initialize output array
    height, width = img.shape
    output = np.zeros((height, width), dtype=np.uint8)

    # Identify strong and weak pixels
    strong_mask = img >= high_thresh
    weak_mask = (img >= low_thresh) & (img < high_thresh)

    # Assign initial pixel values
    output[strong_mask] = strong
    output[weak_mask] = weak

    # Step 3: Find all weak pixel positions
    weak_i, weak_j = np.where(weak_mask)

    # Step 4: Check the neighborhood of each weak pixel
    for i, j in zip(weak_i, weak_j):

        row_start = max(0, i - 1)
        row_end = min(height, i + 2)

        col_start = max(0, j - 1)
        col_end = min(width, j + 2)

        neighborhood = strong_mask[ #3x3 array
            row_start:row_end,
            col_start:col_end
        ]

        # Step 5: Promote or suppress weak pixels
        if np.any(neighborhood):
            output[i, j] = strong
        else:
            output[i, j] = suppressed

    # Step 6: Return binary edge map
    return output

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    # Step 1: Convert to grayscale and smooth the image
    image = Image.open(image_path).convert("L") #L: grayscale
    img = np.asarray(image, dtype=np.float32)

    smoothed = gaussian_filter(img, sigma=1.4)

    # Step 2: Define Sobel kernels
    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    Ky = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ], dtype=np.float32)

    # Calculate gradients
    Gx = convolve(smoothed, Kx)
    Gy = convolve(smoothed, Ky)

    # Step 3: Calculate and normalize gradient magnitude
    magnitude = np.hypot(Gx, Gy)

    max_magnitude = np.max(magnitude)

    if max_magnitude > 0:
        magnitude = magnitude / max_magnitude * 255.0

    # Calculate gradient direction
    angle = np.arctan2(Gy, Gx)

    # Step 4: Apply Non-Maximum Suppression
    thinned = non_maximum_suppression(magnitude, angle)

    # Step 5: Apply hysteresis thresholding
    edges = hysteresis_thresholding(thinned, low, high)

    # Step 6: Save the final edge image
    edge_image = Image.fromarray(edges.astype(np.uint8))
    edge_image.save(output_path)

if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")