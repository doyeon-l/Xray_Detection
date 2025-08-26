import pymysql
import os
import shutil
from ultralytics import YOLO
from datetime import datetime
import argparse  # 인자 파싱을 위해 추가
import sys       # stdout, stderr 리디렉션을 위해 추가
import traceback # 예외 처리를 위해 추가

# --- 🔴 [추가] ---
import requests
import albumentations as A
import cv2
# --- [추가 끝] ---

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
def update_job_status(job_id, status, message=None, log_append=None, version=None, performance=None):
    conn = None
    try:
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME, charset='utf8mb4')
        with conn.cursor() as cursor:
            if log_append:
                # 상태를 'RUNNING'으로 명시적으로 업데이트하며 로그 추가
                cursor.execute("UPDATE retraining_jobs SET status = 'RUNNING', progress_log = CONCAT(IFNULL(progress_log, ''), %s) WHERE id = %s", (log_append, job_id))
            else: # 상태 최종 업데이트 시
                sql = "UPDATE retraining_jobs SET status = %s, result_message = %s, completed_at = NOW()"
                params = [status, message]
                if version:
                    sql += ", version = %s"
                    params.append(version)
                if performance is not None:
                    sql += ", performance = %s"
                    params.append(performance)
                sql += " WHERE id = %s"
                params.append(job_id)
                cursor.execute(sql, tuple(params))
        conn.commit()

        # [수정] 작업 완료/실패 시 이메일 알림 (requests 사용)
        if status in ['COMPLETED', 'FAILED']:
            try:
                # Flask 앱의 API를 호출하여 관리자 이메일 목록을 가져옴
                admin_res = requests.get('http://127.0.0.1:5000/api/admin_emails')
                if admin_res.status_code == 200:
                    recipients = admin_res.json().get('emails', [])
                    if recipients:
                        email_subject = f"[X-Ray 감지 시스템] 모델 재학습 결과: {status}"
                        email_body = f"""
                        <h3>모델 재학습 작업이 종료되었습니다.</h3>
                        <ul>
                            <li><b>작업 ID:</b> {job_id}</li>
                            <li><b>상태:</b> {status}</li>
                            <li><b>결과 메시지:</b> {message}</li>
                            <li><b>종료 시간:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
                        </ul>
                        """
                        # Flask 앱의 이메일 발송 API를 호출
                        requests.post('http://127.0.0.1:5000/api/send_email', json={
                            'subject': email_subject,
                            'recipients': recipients,
                            'body': email_body
                        }, timeout=10)
            except Exception as e:
                # 이메일 발송 실패가 전체 프로세스를 중단시키지 않도록 예외 처리
                print(f"Email notification API call failed: {e}")
    except Exception as e:
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

    # --- [추가] 데이터 증강 파이프라인 정의 ---
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=15, p=0.5),
    ])

    for item in data:
        original_image_path = item['image_path']
        final_class = item['yolo_class']  # '0' for BAD, '1' for GOOD

        if not os.path.exists(original_image_path):
            print(f"경고: 이미지 파일을 찾을 수 없습니다 - {original_image_path}")
            continue

        # --- [수정] 원본 이미지 복사 및 증강 이미지 생성 로직 ---
        # 원본 이미지 로드 (OpenCV 사용)
        image = cv2.imread(original_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Albumentations는 RGB 형식을 사용
        
        # 1. 원본 이미지 저장
        original_basename = os.path.basename(original_image_path)
        cv2.imwrite(os.path.join(IMAGES_PATH, original_basename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # 2. 원본 라벨 파일 생성
        label_file_name = os.path.splitext(original_basename)[0] + '.txt'
        with open(os.path.join(LABELS_PATH, label_file_name), 'w') as f:
            if final_class == '0': # BAD
                # 예시: 이미지 중앙에 고정된 크기의 바운딩 박스
                f.write("0 0.5 0.5 0.5 0.5\n")
            # GOOD 클래스는 빈 파일을 생성

        # 3. 'BAD' 데이터에 대해서만 증강 이미지 2개 추가 생성
        if final_class == '0':
            for i in range(2): # 증강 이미지 2개 생성
                augmented = transform(image=image)
                augmented_image = augmented['image']
                
                aug_basename = f"aug_{i}_{original_basename}"
                save_path = os.path.join(IMAGES_PATH, aug_basename)
                cv2.imwrite(save_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

                # 증강 이미지에 대한 라벨 파일도 생성
                aug_label_name = os.path.splitext(aug_basename)[0] + '.txt'
                with open(os.path.join(LABELS_PATH, aug_label_name), 'w') as f:
                    f.write("0 0.5 0.5 0.5 0.5\n")
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", required=True, type=int)
    args = parser.parse_args()
    job_id = args.job_id

    # stdout과 stderr를 DatabaseLogger로 리디렉션
    sys.stdout = DatabaseLogger(job_id)
    sys.stderr = sys.stdout

    try:
        # 1. 재학습 시작을 DB에 기록
        update_job_status(job_id, 'RUNNING', log_append="재학습 프로세스를 시작합니다...\n")

        # 2. 데이터 가져오기
        reclassified_data = fetch_reclassified_data()
        if not reclassified_data:
            print("재학습할 데이터가 없습니다.\n")
            update_job_status(job_id, 'COMPLETED', message="재학습할 데이터가 없어 작업을 종료합니다.")
            sys.exit()

        # 3. 데이터셋 준비 (증강 포함)
        prepare_dataset(reclassified_data)
        create_yaml_file()
        
        # 4. YOLO 모델 학습 실행
        print("YOLO 모델 재학습 시작...\n")
        model = YOLO('model/best.pt')  # 기존 모델을 불러와서 fine-tuning
        results = model.train(
            data=DATA_YAML_PATH,
            epochs=50,
            imgsz=640,
            name=f'retrain_{datetime.now().strftime("%Y%m%d_%H%M")}'
        )
        
        # 5. 학습 결과 분석 및 버전 관리
        # YOLOv8의 results 객체에서 최종 성능 지표(mAP50-95) 추출
        final_map_score = results.results_dict.get('metrics/mAP50-95(B)', 0.0)
        
        # 새로운 버전 번호 생성
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # version이 NULL이 아닌 마지막 버전을 가져옴
            cursor.execute("SELECT version FROM retraining_jobs WHERE version IS NOT NULL ORDER BY id DESC LIMIT 1")
            last_version_row = cursor.fetchone()
            if last_version_row and last_version_row[0]:
                major, minor = map(int, last_version_row[0].replace('v', '').split('.'))
                new_version_str = f'v{major}.{minor + 1}'
            else:
                new_version_str = 'v1.1' # 첫 재학습일 경우
        conn.close()
        
        # 6. 최종 결과를 DB에 업데이트
        print("\n모델 재학습이 성공적으로 완료되었습니다.")
        update_job_status(
            job_id, 'COMPLETED',
            message=f"재학습 완료 (mAP: {final_map_score:.4f})",
            version=new_version_str,
            performance=final_map_score
        )

    except Exception as e:
        # 오류 발생 시 DB에 'FAILED'로 기록
        error_details = traceback.format_exc()
        print(f"\n오류 발생: {e}\n{error_details}")
        update_job_status(job_id, 'FAILED', message=f"오류 발생: {e}")