import pymysql
import os
import shutil
from ultralytics import YOLO
from datetime import datetime
import argparse  # 인자 파싱을 위해 추가
import sys       # stdout, stderr 리디렉션을 위해 추가
import traceback # 예외 처리를 위해 추가

# ----- 설정 -----
DB_HOST = '127.0.0.1'
DB_USER = 'root'
DB_PASSWORD = 'root123'
DB_NAME = 'mysql'

# 재학습용 데이터셋을 구성할 폴더
RETRAIN_DATASET_PATH = './retrain_dataset'
IMAGES_PATH = os.path.join(RETRAIN_DATASET_PATH, 'images/train')
LABELS_PATH = os.path.join(RETRAIN_DATASET_PATH, 'labels/train')

# 데이터셋 설정 파일 (YAML) 경로
DATA_YAML_PATH = os.path.join(RETRAIN_DATASET_PATH, 'data.yaml')

# DB 연결 및 업데이트 함수
def get_db_connection():
    return pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME, charset='utf8mb4', autocommit=True)

# DB 업데이트 함수가 자체적으로 연결을 관리하도록 변경
def update_job_status(job_id, status, message=None, log_append=None):
    conn = None
    try:
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME, charset='utf8mb4')
        with conn.cursor() as cursor:
            if log_append:
                # 상태를 'RUNNING'으로 명시적으로 업데이트
                cursor.execute("UPDATE retraining_jobs SET status = 'RUNNING', progress_log = CONCAT(IFNULL(progress_log, ''), %s) WHERE id = %s", (log_append, job_id))
            elif message:
                cursor.execute("UPDATE retraining_jobs SET status = %s, result_message = %s, completed_at = NOW() WHERE id = %s", (status, message, job_id))
            else:
                cursor.execute("UPDATE retraining_jobs SET status = %s WHERE id = %s", (status, job_id))
        conn.commit()
    except Exception as e:
        # 이 함수 자체에서 오류가 나면 터미널에만 출력
        print(f"DB update failed: {e}")
    finally:
        if conn:
            conn.close()

class DatabaseLogger:
    def __init__(self, job_id):
        self.terminal = sys.stdout
        self.job_id = job_id

    def write(self, message):
        self.terminal.write(message)
        # 쓸 때마다 DB에 업데이트 함수를 호출
        update_job_status(self.job_id, 'RUNNING', log_append=message)

    def flush(self):
        pass

# ----- 1. 데이터베이스 연결 및 재분류된 데이터 가져오기 -----
def fetch_reclassified_data():
    print("데이터베이스에서 재분류된 데이터를 가져옵니다...")
    conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    with conn.cursor() as cursor:
        cursor.execute("SELECT image_path, yolo_class FROM classified_objects WHERE is_reclassified = 1 AND del_yn = 'N'")
        data = cursor.fetchall()
    conn.close()
    print(f"총 {len(data)}개의 재분류된 데이터를 찾았습니다.")
    return data

# ----- 2. 학습용 데이터셋 구성 -----
def prepare_dataset(data):
    print("재학습용 데이터셋을 구성합니다...")
    # 기존 폴더가 있다면 삭제 후 다시 생성
    if os.path.exists(RETRAIN_DATASET_PATH):
        shutil.rmtree(RETRAIN_DATASET_PATH)
    os.makedirs(IMAGES_PATH)
    os.makedirs(LABELS_PATH)

    for item in data:
        original_image_path = item['image_path']
        final_class = item['yolo_class']  # '0' for BAD, '1' for GOOD

        if not os.path.exists(original_image_path):
            print(f"경고: 이미지 파일을 찾을 수 없습니다 - {original_image_path}")
            continue

        # 이미지 파일을 새 경로로 복사
        new_image_name = os.path.basename(original_image_path)
        shutil.copy(original_image_path, os.path.join(IMAGES_PATH, new_image_name))

        # 라벨 파일 생성
        label_file_name = os.path.splitext(new_image_name)[0] + '.txt'
        label_file_path = os.path.join(LABELS_PATH, label_file_name)

        # 'BAD' (class 0)일 경우에만 바운딩 박스 정보가 있다고 가정
        # 이 부분은 실제 YOLO 라벨링 정보가 DB에 있거나, 다시 추론해야 함.
        # 여기서는 단순화를 위해 BAD일 경우 0.5 0.5 0.5 0.5 로 임의의 박스를 생성
        with open(label_file_path, 'w') as f:
            if final_class == '0': # BAD
                # 실제로는 원본 YOLO 라벨을 가져와야 함.
                # 여기서는 예시로 이미지 중앙에 박스를 그린다.
                f.write("0 0.5 0.5 0.5 0.5\n")
            # GOOD (class 1)일 경우, 빈 파일을 생성

    print("데이터셋 구성 완료.")

# ----- 3. YAML 파일 생성 -----
def create_yaml_file():
    print("data.yaml 파일을 생성합니다...")
    yaml_content = f"""
        train: {os.path.abspath(IMAGES_PATH)}
        val: {os.path.abspath(IMAGES_PATH)}  # 간단하게 train set을 val로도 사용

        nc: 2
        names: ['BAD', 'GOOD']
        """
    with open(DATA_YAML_PATH, 'w') as f:
        f.write(yaml_content)
    print("YAML 파일 생성 완료.")


# ----- 4. YOLO 모델 학습 실행 -----
def run_training():
    print("YOLO 모델 재학습을 시작합니다...")
    # 기존 best.pt를 가중치로 사용하여 fine-tuning 시작
    model = YOLO('model/best.pt') 

    # 학습 실행
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=50,
        imgsz=640,
        name=f'retrain_{datetime.now().strftime("%Y%m%d_%H%M")}'
    )
    
    print("모델 재학습이 완료되었습니다.")
    # (선택) 학습이 끝난 후 새로운 best.pt를 원래 위치로 복사
    # best_model_path = results.save_dir / 'weights/best.pt'
    # shutil.copy(best_model_path, './model/best_retrained.pt')
    # print(f"새로운 모델이 './model/best_retrained.pt'에 저장되었습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", required=True, type=int)
    args = parser.parse_args()
    job_id = args.job_id

    sys.stdout = DatabaseLogger(job_id)
    sys.stderr = sys.stdout

    try:
        # 시작 상태를 명시적으로 업데이트
        update_job_status(job_id, 'RUNNING', log_append="재학습 프로세스를 시작합니다...\n")

        print("1. 재분류된 데이터 가져오기...\n")
        reclassified_data = fetch_reclassified_data()
        if not reclassified_data:
            print("재학습할 데이터가 없습니다.\n")
            update_job_status(job_id, 'COMPLETED', message="재학습할 데이터가 없어 작업을 종료합니다.")
            sys.exit()

        print(f"총 {len(reclassified_data)}개의 데이터로 학습을 준비합니다.\n")
        print("2. 학습용 데이터셋 구성 중...\n")
        prepare_dataset(reclassified_data)
        create_yaml_file()
        print("데이터셋 구성 완료.\n")
        print("3. YOLO 모델 재학습 시작...\n")

        model = YOLO('model/best.pt')
        results = model.train(data=DATA_YAML_PATH, epochs=50, imgsz=640, name=f'retrain_{datetime.now().strftime("%Y%m%d_%H%M")}')

        print("\n모델 재학습이 성공적으로 완료되었습니다.")
        update_job_status(job_id, 'COMPLETED', message="모델 재학습이 성공적으로 완료되었습니다.")

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"\n오류 발생: {e}\n{error_details}")
        update_job_status(job_id, 'FAILED', message=f"오류 발생: {e}")