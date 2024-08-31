import cv2
import subprocess
import os
import serial
import time

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=6,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def capture_image(cap, filename='/home/aieessu/yolov5/data/images/captured_image.jpg'):
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        return False

    cv2.imwrite(filename, frame)
    print("Image captured successfully")
    return True

def delete_files(image_path='/home/aieessu/yolov5/data/images/captured_image.jpg', exp_folder='/home/aieessu/yolov5/runs/detect/exp'):
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Deleted file: {image_path}")
    else:
        print(f"File {image_path} does not exist. Skipping deletion.")

    if os.path.exists(exp_folder) and os.path.isdir(exp_folder):
        subprocess.run(['rm', '-rf', exp_folder])
        print(f"Deleted folder: {exp_folder}")
    else:
        print(f"Folder {exp_folder} does not exist. Skipping deletion.")

def run_detection(image_path='/home/aieessu/yolov5/data/images/captured_image.jpg'):
    result = subprocess.run([
        'python3', '/home/aieessu/yolov5/detect.py',
        '--weights', '/home/aieessu/yolov5/runs/train/ecoffee11/weights/best.pt',
        '--source', image_path,
        '--project', '/home/aieessu/yolov5/runs/detect',
        '--save-txt',
        '--conf', '0.5',
    ], capture_output=True, text=True)

    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    img_width = 1920
    img_height = 1080

    result_file = '/home/aieessu/yolov5/runs/detect/exp/labels/' + os.path.basename(image_path).replace('.jpg', '.txt')

    if not os.path.isfile(result_file):
        print("It's not a plastic cup")
        return "0"

    with open(result_file, 'r') as file:
        lines = file.readlines()

    class_counts = {}
    real_height = None

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        bbox = list(map(float, parts[1:5]))

        class_name = str(class_id)
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1

        if class_id == 0:
            real_height = bbox[3]
            break

    if '0' in class_counts and class_counts['0'] > 0:
        print(f"Cup Height: {real_height}")
        if '4' in class_counts and class_counts['0'] > 0:
            print("detect logo cup, you get 4 points")
            return "1"
        else:
            if real_height > 0.57:
                print("clean L cup, you get 10 points") 
                return "2"
            else:
                print("clean M cup, you get 8 points")
                return "3"
    else:
        print("It's not a plastic cup")
        return "0"

if __name__ == "__main__":
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Unable to open camera")
        exit()

    py_serial = serial.Serial(
        port='/dev/ttyACM0',
        baudrate=9600,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera")
            break

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            delete_files()
            if capture_image(cap):
                result = run_detection()
                print(result)
               
                # YOLOv5가 처리한 이미지를 읽어서 화면에 표시
                detected_image_path = '/home/aieessu/yolov5/runs/detect/exp/' + os.path.basename('/home/aieessu/yolov5/data/images/captured_image.jpg')
                detected_image = cv2.imread(detected_image_path)
               
                if detected_image is not None:
                    cv2.imshow("Detected Image", detected_image)
                else:
                    print("Failed to load detected image.")
               
                # 결과를 시리얼로 전송
                py_serial.write(result.encode())
               
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    py_serial.close()