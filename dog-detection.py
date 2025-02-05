import mediapipe as mp
from mediapipe.tasks import python
import cv2
import matplotlib.pyplot as plt


model_path = 'model_path' # model file goes here

# Load the input image from an image file.
numpy_image = cv2.imread('img1.jpg')  #

# Convert the image from BGR (OpenCV default) to RGB.
# numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

# Load the input image from a numpy array.
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=5,
    running_mode=VisionRunningMode.IMAGE)

with ObjectDetector.create_from_options(options) as detector:
    detection_result = detector.detect(mp_image)
    # print(detection_result)
    # Draw bounding boxes on the image for each detected object.
    for detection in detection_result.detections:
        # Get bounding box and convert it to pixel coordinates.
        # print(detection)

        bbox = detection.bounding_box
        x = int(bbox.origin_x)
        y = int(bbox.origin_y)
        w = int(bbox.width)
        h = int(bbox.height)

        # Draw a rectangle around the detected object.
        cv2.rectangle(numpy_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # # Show the image with the bounding boxes.
    # cv2.imshow('Object Detection', numpy_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # Convert the image to RGB for displaying with matplotlib.
    numpy_image_rgb_display = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib (numplot).
    plt.imshow(numpy_image_rgb_display)
    plt.axis('off')  # Hide axis
    plt.show()