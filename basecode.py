import cv2
import subprocess
import json
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
    flip_method=0,
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

def capture_image(filename='/home/aieessu/yolov5/data/images/captured_image.jpg'):
    pipeline = gstreamer_pipeline()
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("unable to open camera")

    ret, frame = cap.read()
    if not ret:
        print("fail to capture image")

    cv2.imwrite(filename, frame)
    print("success capture")
    cap.release()


def delete_files(image_path='/home/aieessu/yolov5/data/images/captured_image.jpg', exp_folder='/home/aieessu/yolov5/runs/detect/exp'):
    # 이미지 파일 삭제
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Deleted file: {image_path}")
    else:
        print(f"File {image_path} does not exist. Skipping deletion.")

    # exp 폴더 삭제
    if os.path.exists(exp_folder) and os.path.isdir(exp_folder):
        subprocess.run(['rm', '-rf', exp_folder])
        print(f"Deleted folder: {exp_folder}")
    else:
        print(f"Folder {exp_folder} does not exist. Skipping deletion.")

def run_detection(image_path='/home/aieessu/yolov5/data/images/captured_image.jpg', output_path='result.txt'):
    result = subprocess.run([
        'python3', '/home/aieessu/yolov5/detect.py',
        '--weights', '/home/aieessu/yolov5/runs/train/ecoffee/weights/best.pt',
        '--source', image_path, #'/home/aieessu/yolov5/data/images/cup_image2.jpg',  # image_path,
        '--project', '/home/aieessu/yolov5/runs/detect',
        '--save-txt',
        '--conf', '0.3',
    ], capture_output=True, text=True)

    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    result_file = '/home/aieessu/yolov5/runs/detect/exp/labels/' + os.path.basename(image_path).replace('.jpg', '.txt')

    if not os.path.isfile(result_file):
        #raise Exception(f"결과 파일 {result_file}을 찾을 수 없습니다.")
        print("It's not plastic cup")
        return 0
    with open(result_file, 'r') as file:
        lines = file.readlines()

    class_counts = {}
    bboxes = []

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        bbox = list(map(float, parts[1:5]))

        class_name = str(class_id)
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1

        bboxes.append(bbox)

    bbox_sizes = [((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) for bbox in bboxes]

    results = {
        'class_counts': class_counts,
        'bbox_sizes': bbox_sizes,
    }

    print("Results Dictionary:")
    for key, value in results.items():
        print(f"{key}: {value}")

    with open(output_path, 'w') as file:
        json.dump(results, file, indent=4)

    # 0번 클래스가 감지되었는지 확인하고, 감지되었다면 1을 리턴합니다.
    if '4' in class_counts and class_counts['0'] > 0:
        print("recognize logo, you can't get points")
        return 1
    elif '0' in class_counts and class_counts['0'] > 0 :
        print("clean cup, you get points")
        return 2
    else:
        print("It's not platsic cup")
        return 0

if __name__ == "__main__":
    delete_files()
    capture_image()
    result = run_detection()
    print(result)

    py_serial = serial.Serial(
        port='/dev/ttyACM0',
        baudrate = 9600,
    )

    while True :
        cmd = result
        py_serial.write(cmd.encode())
        time.sleep(0.1)

        if py_serial.readable():
            response = py_serial.readline()
            print(response[:len(response)-1].decode())


