import sys
from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import pymysql
from datetime import datetime, timedelta
import csv
import io
import cv2
import uuid
import os
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as nn
from PIL import Image
from ultralytics import YOLO
from pytorch_msssim import ms_ssim # pip install pytorch-msssim
from efficientnet_pytorch import EfficientNet  # pip install efficientnet_pytorch
from model.models import EfficientNetAutoencoder # 👈 직접 작성한 모델 클래스 import 필요
from model.classifier import EfficientNetClassifier
# from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from model.models import MSSSIMLoss
from functools import wraps
from flask import abort
import subprocess  # 👈 [기능 3] 재학습 스크립트 실행을 위해 추가

# 👈 [기능 2] XAI (Grad-CAM) 라이브러리 추가
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.grad_cam import GradCAM
import ttach as tta

import psutil # 🚀 프로세스 제어를 위해 psutil 라이브러리가 필요합니다 (pip install psutil)

app = Flask(__name__)
uploadPath = './static/upload'
modelPath = './model'
xaiResultPath = './static/xai_results' # 👈 [기능 2] XAI 결과 저장 폴더

# XAI 결과 폴더가 없으면 생성
if not os.path.exists(xaiResultPath):
    os.makedirs(xaiResultPath)

app.secret_key = 'your-secret-key-for-fubao-project'
app.config['ADMIN_SECRET_CODE'] = 'admin123'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

class User(UserMixin):
    def __init__(self, id, userid, password_hash, name, email, company, role, is_admin):
        self.id = id
        self.username = userid
        self.password_hash = password_hash
        self.name = name
        self.email = email
        self.company = company
        self.role = role
        self.is_admin = is_admin

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT id, userid, password_hash, name, email, company, role, is_admin FROM users WHERE id = %s", (user_id,))
        user_data = cursor.fetchone()
    conn.close()
    if user_data:
        return User(id=user_data['id'], userid=user_data['userid'], password_hash=user_data['password_hash'], 
                    name=user_data['name'], email=user_data['email'], 
                    company=user_data['company'], role=user_data['role'], is_admin=user_data['is_admin'])
    return None

# --- 모델 로드 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 지도학습 분류 모델 (EfficientNet-B3)
classifier_model = EfficientNetClassifier(num_classes=2).to(device)
classifier_model.load_state_dict(torch.load(os.path.join(modelPath, 'eff_from_yolo_infer.pth'), map_location=device))
classifier_model.eval()

# 비지도학습 이상 탐지 모델 (Autoencoder with EfficientNet-B2)
autoencoder_model = EfficientNetAutoencoder(model_version='b2', output_size=224).to(device)
autoencoder_model.load_state_dict(torch.load(os.path.join(modelPath, 'autoencoder_effnetb2_img224_batch16_epoch100_M80_SS20.pth'), map_location=device))
autoencoder_model.eval()

# YOLO 모델
yolo_model = YOLO(os.path.join(modelPath, 'best.pt'))

def get_db_connection():
    return pymysql.connect(
        host='127.0.0.1', user='root', password='root123', db='mysql',
        charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor
    )

def get_transform(size=300): # B3 기준 300
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_with_classifier(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class_index = torch.argmax(probabilities).item()
        score = probabilities[0, predicted_class_index].item() # 예측된 클래스의 확률을 점수로 사용
    return "GOOD" if predicted_class_index == 1 else "BAD", score

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# --- [수정] 업로드 및 추론 라우트 ---
@app.route('/upload', methods=['POST'])
@login_required
def upload_files():
    std_date = request.form.get('std_date')
    model_gb = request.form.get('model_gb', 'S')  # 'S' 또는 'U'
    files = request.files.getlist('files')
    results = []

    if not std_date or not files:
        return jsonify({'status': 'error', 'message': '기준일과 파일이 필요합니다.'}), 400

    for file in files:
        if file and allowed_file(file.filename):
            org_image_name = file.filename
            filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8] + '.jpg'
            filepath = os.path.join(uploadPath, filename)
            file.save(filepath)

            img_pil = Image.open(filepath).convert('RGB')

            initial_prediction, yolo_class, effnet_class = 'UNKNOWN', '0', 'UNKNOWN'
            score, anomaly_score = 0.0, None

            if model_gb == 'S':
                # 🚀 [핵심 수정] YOLO 객체 탐지 모델의 결과 처리 로직으로 변경
                yolo_results = yolo_model.predict(source=filepath, verbose=False)

                # 탐지된 객체(Box)의 개수를 확인합니다.
                num_detections = len(yolo_results[0].boxes)

                # 💡 [핵심 수정] 신뢰도 임계값 변수 추가
                confidence_threshold = 0.5 # 50% 신뢰도

                # 💡 [진단 코드 추가] 터미널(콘솔)에서 탐지 결과를 확인합니다.
                print(f"--- [Debug] Image: {org_image_name} ---")
                print(f"Detections found: {num_detections}")
                if num_detections > 0:
                    top_confidence = yolo_results[0].boxes.conf[0].item()
                    print(f"Top detection confidence: {top_confidence:.4f}")
                print("-------------------------------------------")

                # 💡 [핵심 수정] 탐지된 객체가 있고, 그 신뢰도가 임계값보다 높은 경우에만 BAD로 판정
                if num_detections > 0 and yolo_results[0].boxes.conf[0].item() > confidence_threshold:
                    initial_prediction = "BAD"
                    yolo_class = '0'
                    score = yolo_results[0].boxes.conf[0].item()
                else:
                    # 탐지된 것이 없거나, 신뢰도가 너무 낮으면 GOOD으로 판정
                    initial_prediction = "GOOD"
                    yolo_class = '1'
                    # 점수는 탐지 결과에 따라 다르게 설정 (없으면 1.0, 낮으면 해당 점수)
                    score = yolo_results[0].boxes.conf[0].item() if num_detections > 0 else 1.0

                effnet_class = initial_prediction

            elif model_gb == 'U':
                # 비지도학습 로직은 기존과 동일합니다.
                transform = get_transform(size=224)
                input_tensor = transform(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    reconstructed = autoencoder_model(input_tensor)
                    reconstruction_error = nn.mse_loss(reconstructed, input_tensor).item()

                anomaly_score = reconstruction_error
                # 💡 이상 점수 임계값은 모델 성능에 따라 조정이 필요할 수 있습니다.
                threshold = 0.6
                initial_prediction = "GOOD" if anomaly_score < threshold else "BAD"
                yolo_class = '1' if initial_prediction == 'GOOD' else '0'
                effnet_class = initial_prediction
                # 점수는 (1 - 이상 점수)로 변환하여 0~1 사이 값으로 표시
                score = max(0.0, 1.0 - anomaly_score)

                # 💡 [진단 코드 추가] 터미널(콘솔)에서 비지도학습 탐지 결과를 확인합니다.
                print(f"--- [Debug Unsupervised] Image: {org_image_name} ---")
                print(f"Anomaly Score (Reconstruction Error): {anomaly_score:.4f}")
                print(f"Threshold: {threshold}")
                print(f"Final Prediction: {initial_prediction}")
                print("----------------------------------------------------")

            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO classified_objects
                        (std_date, model_gb, image_path, image_name, org_image_name, yolo_class, effnet_class, score, anomaly_score, initial_prediction)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (std_date, model_gb, filepath, filename, org_image_name, yolo_class, effnet_class, score, anomaly_score, initial_prediction))
                conn.commit()
            conn.close()

            results.append({'filename': filename, 'effnet_class': effnet_class})

    return jsonify({'status': 'success', 'results': results})

# --- 🚀 [신규] XAI (Grad-CAM) 생성 API ---
@app.route('/api/grad_cam/<int:item_id>', methods=['GET'])
@login_required
def generate_grad_cam(item_id):
    conn = get_db_connection()
    with conn.cursor() as cursor:
        # 💡 [수정] 이제 model_gb 컬럼도 함께 조회합니다.
        cursor.execute("SELECT image_path, xai_image_path, model_gb FROM classified_objects WHERE id = %s", (item_id,))
        item = cursor.fetchone()
    conn.close()

    if not item:
        return jsonify({'status': 'error', 'message': '이미지를 찾을 수 없습니다.'}), 404

    # XAI 이미지가 이미 있다면 바로 반환 (캐싱)
    if item['xai_image_path']:
        return jsonify({'status': 'success', 'xai_path': item['xai_image_path']})

    try:
        image_path = item['image_path']
        img_pil = Image.open(image_path).convert('RGB')
        
        visualization = None # 시각화 결과를 담을 변수

        # 💡 [핵심] 모델 구분에 따라 다른 XAI 로직을 실행
        if item['model_gb'] == 'S':
            # --- 1. 지도학습 모델: Grad-CAM (기존 로직) ---
            img_pil_resized = img_pil.resize((300, 300))
            rgb_img = np.array(img_pil_resized, dtype=np.float32) / 255
            transform = get_transform(size=300)
            input_tensor = transform(img_pil_resized).unsqueeze(0).to(device)

            target_layers = [classifier_model.features[-1]]
            cam = GradCAM(model=classifier_model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        elif item['model_gb'] == 'U':
            # --- 2. 비지도학습 모델: 복원 오차 맵 (신규 로직) ---
            transform = get_transform(size=224)
            input_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                reconstructed_tensor = autoencoder_model(input_tensor)

            # 텐서를 시각화 가능한 이미지(numpy 배열)로 변환
            original_img_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            reconstructed_img_np = reconstructed_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            
            # 픽셀 단위로 차이를 계산 (오차 맵)
            error_map = np.abs(original_img_np - reconstructed_img_np)
            error_map_gray = np.mean(error_map, axis=2) # 흑백으로 변환
            
            # 히트맵 생성
            heatmap = cv2.normalize(error_map_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # 원본 이미지도 0~255 범위의 uint8 타입으로 변환
            original_img_display = cv2.normalize(original_img_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # 원본 이미지와 히트맵을 합성
            superimposed_img = cv2.addWeighted(heatmap, 0.5, original_img_display, 0.5, 0)
            visualization = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB) # PIL 저장을 위해 RGB로 변환

        if visualization is not None:
            xai_filename = f"xai_{os.path.basename(image_path)}"
            xai_filepath = os.path.join(xaiResultPath, xai_filename)
            Image.fromarray(visualization).save(xai_filepath)

            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("UPDATE classified_objects SET xai_image_path = %s WHERE id = %s", (xai_filepath, item_id))
                conn.commit()
            conn.close()

            return jsonify({'status': 'success', 'xai_path': xai_filepath})
        else:
             return jsonify({'status': 'error', 'message': '해당 모델에 대한 XAI를 생성할 수 없습니다.'}), 500

    except Exception as e:
        print(f"XAI 이미지 생성 중 오류 발생: {e}")
        return jsonify({'status': 'error', 'message': f'XAI 이미지 생성 중 오류가 발생했습니다: {e}'}), 500


# 🚀 [신규] 성능 모니터링 API ---
@app.route('/stats/performance_trend')
@login_required
def stats_performance_trend():
    query = """
        SELECT
            YEARWEEK(created_at, 1) AS year_week,
            COUNT(id) AS total_count,
            SUM(CASE WHEN initial_prediction = 'GOOD' AND yolo_class = '1' THEN 1
                     WHEN initial_prediction = 'BAD' AND yolo_class = '0' THEN 1
                     ELSE 0 END) AS correct_count,
            SUM(CASE WHEN yolo_class = '0' THEN 1 ELSE 0 END) AS actual_bad,
            SUM(CASE WHEN initial_prediction = 'BAD' AND yolo_class = '0' THEN 1 ELSE 0 END) AS true_positives
        FROM classified_objects
        WHERE del_yn = 'N'
        GROUP BY year_week
        ORDER BY year_week DESC
        LIMIT 8;
    """
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(query)
        data = cursor.fetchall()
        
        # 💡 [핵심 수정] fetchall()이 반환하는 튜플(tuple)을 리스트(list)로 변환합니다.
        data = list(data)

        for i, row in enumerate(data):
            total = row['total_count']
            correct = row['correct_count']
            actual_bad = row['actual_bad']
            tp = row['true_positives']

            row['accuracy'] = round((correct / total * 100) if total > 0 else 0, 2)
            row['recall'] = round((tp / actual_bad * 100) if actual_bad > 0 else 0, 2)
            row['week_label'] = f"{- (len(data) - 1 - i)}주"

    conn.close()
    data.reverse()
    return jsonify(data)
    
# --- 🚀 [신규] 모델 재학습 트리거 API ---
@app.route('/admin/retrain_model', methods=['POST'])
@admin_required
def retrain_model():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 💡 [수정] 이미 실행 중인 작업이 있는지 확인
            cursor.execute("SELECT id FROM retraining_jobs WHERE status = 'RUNNING' OR status = 'PENDING'")
            if cursor.fetchone():
                flash('이미 재학습 작업이 진행 중입니다.', 'warning')
                return redirect(url_for('model_management'))

            cursor.execute("INSERT INTO retraining_jobs (status, progress_log) VALUES ('PENDING', '재학습 작업을 대기열에 추가했습니다...\\n')")
            conn.commit()
            job_id = cursor.lastrowid

            # train.py를 백그라운드 프로세스로 실행
            process = subprocess.Popen([sys.executable, 'train.py', '--job_id', str(job_id)])
            
            # 💡 [신규] 생성된 프로세스의 PID를 DB에 즉시 저장
            cursor.execute("UPDATE retraining_jobs SET process_id = %s WHERE id = %s", (process.pid, job_id))
            conn.commit()

        flash('모델 재학습 프로세스가 시작되었습니다.', 'success')
    except Exception as e:
        flash(f'재학습 시작 중 오류 발생: {e}', 'error')
    finally:
        if conn:
            conn.close()
    return redirect(url_for('model_management'))


# --- 🚀 [신규] 재학습 중지 API ---
@app.route('/api/stop_retraining', methods=['POST'])
@admin_required
def stop_retraining_job():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 현재 실행중인 작업의 PID를 찾음
            cursor.execute("SELECT id, process_id FROM retraining_jobs WHERE status = 'RUNNING' ORDER BY id DESC LIMIT 1")
            job = cursor.fetchone()

            if job and job.get('process_id'):
                pid = job['process_id']
                job_id = job['id']
                try:
                    # psutil을 사용하여 해당 프로세스 종료
                    p = psutil.Process(pid)
                    p.terminate() # 프로세스 강제 종료
                    message = "사용자에 의해 작업이 취소되었습니다."
                    # DB 상태를 'CANCELED'로 업데이트
                    cursor.execute("UPDATE retraining_jobs SET status = 'CANCELED', result_message = %s, completed_at = NOW() WHERE id = %s", (message, job_id))
                    conn.commit()
                    return jsonify({'status': 'success', 'message': message})
                except psutil.NoSuchProcess:
                    message = "프로세스를 찾을 수 없지만, 작업을 취소됨으로 처리합니다."
                    cursor.execute("UPDATE retraining_jobs SET status = 'CANCELED', result_message = %s, completed_at = NOW() WHERE id = %s", (message, job_id))
                    conn.commit()
                    return jsonify({'status': 'warning', 'message': message})
            else:
                return jsonify({'status': 'error', 'message': '중지할 수 있는 실행 중인 작업이 없습니다.'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if conn:
            conn.close()

# --- 🚀 [신규] 재학습 상태 확인 API ---
@app.route('/api/retraining_status')
@admin_required
def get_retraining_status():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        # 가장 최근의 작업 1개만 조회
        cursor.execute("SELECT * FROM retraining_jobs ORDER BY id DESC LIMIT 1")
        job = cursor.fetchone()
    conn.close()

    if job:
        # 날짜/시간 객체를 문자열로 변환 (JSON으로 보내기 위해)
        for key, value in job.items():
            if isinstance(value, datetime):
                job[key] = value.strftime('%Y-%m-%d %H:%M:%S')
        return jsonify(job)
    else:
        # 아직 아무 작업도 없는 경우
        return jsonify({'status': 'NO_JOB', 'progress_log': '아직 재학습 작업이 시작되지 않았습니다.'})

@app.route('/')
def index():
    return render_template('index.html')

# --- list.html 렌더링 ---
@app.route('/list')
@login_required # 👈 이제 목록 페이지는 로그인이 필요합니다.
def list_page():
    return render_template('list.html')

@app.route('/api/list', methods=['GET'])
def api_list():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('limit', 20))
    offset = (page - 1) * per_page
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    yolo_class = request.args.get('yolo_class')
    search_term = request.args.get('search_term')
    model_gb = request.args.get('model_gb', 'S')
    sort_by = request.args.get('sort_by', 'id')
    sort_order = request.args.get('sort_order', 'DESC')

    base_query = "FROM classified_objects WHERE del_yn = 'N'"
    params = []

    if model_gb in ('S', 'U'):
        base_query += " AND model_gb = %s"
        params.append(model_gb)
    if from_date:
        base_query += " AND std_date >= %s"
        params.append(from_date)
    if to_date:
        base_query += " AND std_date <= %s"
        params.append(to_date)
    if yolo_class in ('0', '1'):
        base_query += " AND yolo_class = %s"
        params.append(yolo_class)
    if search_term:
        base_query += " AND org_image_name LIKE %s"
        params.append(f"%{search_term}%")

    count_query = "SELECT COUNT(*) as total " + base_query
    
    allowed_sort_columns = ['id', 'std_date', 'org_image_name', 'yolo_class', 'created_at', 'anomaly_score']
    order_clause = f" ORDER BY {sort_by} {sort_order.upper()}" if sort_by in allowed_sort_columns and sort_order.upper() in ['ASC', 'DESC'] else " ORDER BY id DESC"
        
    data_query = "SELECT id, std_date, model_gb, image_path, image_name, org_image_name, yolo_class, effnet_class, score, anomaly_score, DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i:%%s') AS created_at, note, is_reclassified, modified_by, IFNULL(DATE_FORMAT(modified_at, '%%Y-%%m-%%d %%H:%%i:%%s'), '') AS modified_at " + base_query + order_clause + " LIMIT %s OFFSET %s"
    
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(count_query, tuple(params))
        total_count = cursor.fetchone()['total']
        data_params = tuple(params + [per_page, offset])
        cursor.execute(data_query, data_params)
        result_data = cursor.fetchall()
    conn.close()

    return jsonify({'total': total_count, 'data': result_data})

@app.route('/api/export', methods=['GET'])
def export_csv():
    # api_list와 동일한 로직으로 데이터를 가져옵니다.
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    yolo_class = request.args.get('yolo_class')
    search_term = request.args.get('search_term')
    sort_by = request.args.get('sort_by', 'id')
    sort_order = request.args.get('sort_order', 'DESC')

    query = "SELECT id, std_date, org_image_name, IF(yolo_class='1', 'GOOD', 'BAD') as status, created_at, note, IF(is_reclassified=1, 'Yes', 'No') as reclassified FROM classified_objects WHERE del_yn = 'N'"
    params = []
    if from_date: query += " AND std_date >= %s"; params.append(from_date)
    if to_date: query += " AND std_date <= %s"; params.append(to_date)
    if yolo_class in ('0', '1'): query += " AND yolo_class = %s"; params.append(yolo_class)
    if search_term: query += " AND org_image_name LIKE %s"; params.append(f"%{search_term}%")
    
    allowed_sort_columns = ['id', 'std_date', 'org_image_name', 'yolo_class', 'created_at']
    if sort_by in allowed_sort_columns and sort_order.upper() in ['ASC', 'DESC']:
        query += f" ORDER BY {sort_by} {sort_order.upper()}"
    else:
        query += " ORDER BY id DESC"
    
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(query, params)
        data = cursor.fetchall()
    conn.close()

    # CSV 파일 생성
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', '검출 기준일', '파일명', '상태', '업로드 일시', '메모', '재분류 여부'])
    for row in data:
        writer.writerow([row['id'], row['std_date'], row['org_image_name'], row['status'], row['created_at'], row['note'], row['reclassified']])
    
    output.seek(0)

    # 💡 수정된 부분: 문자열을 'utf-8-sig'로 인코딩하여 바이트로 만듭니다.
    # 이렇게 하면 파일 시작 부분에 BOM이 추가되어 Excel이 한글을 올바르게 인식합니다.
    csv_data = output.getvalue().encode('utf-8-sig')

    response = make_response(csv_data)
    response.headers["Content-Disposition"] = f"attachment; filename=export_{datetime.now().strftime('%Y%m%d')}.csv"
    response.headers["Content-type"] = "text/csv; charset=utf-8-sig"
    return response

@app.route('/api/delete', methods=['POST'])
def api_delete():
    ids = request.json.get('ids', [])
    if not ids: return jsonify({'status': 'no_ids'}), 400
    query = "UPDATE classified_objects SET del_yn = 'Y' WHERE id IN (%s)" % ','.join(['%s'] * len(ids))
    conn = get_db_connection()
    with conn.cursor() as cursor: cursor.execute(query, ids)
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

# --- 재분류 API ---
@app.route('/api/reclassify', methods=['POST'])
@login_required
def api_reclassify():
    # ... (이 함수는 그대로 유지) ...
    data = request.json
    item_id = data.get('id')
    new_class = data.get('new_class')
    modifier = current_user.username

    if not item_id or new_class not in ('0', '1'):
        return jsonify({'status': 'error', 'message': 'Invalid data'}), 400

    query = """
        UPDATE classified_objects
        SET
            yolo_class = %s, effnet_class = %s,
            is_reclassified = 1,
            modified_at = NOW(), modified_by = %s
        WHERE id = %s
    """
    # effnet_class도 동일하게 업데이트, is_reclassified는 1로 고정
    new_effnet_class = 'GOOD' if new_class == '1' else 'BAD'
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(query, (new_class, new_effnet_class, modifier, item_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})


# 🚀 [신규] 선택 항목 일괄 재분류 API ---
@app.route('/api/reclassify_batch', methods=['POST'])
@login_required
def api_reclassify_batch():
    items = request.json.get('items', [])
    if not items:
        return jsonify({'status': 'error', 'message': 'No items selected'}), 400

    modifier = current_user.username
    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            for item in items:
                item_id = item.get('id')
                current_class = item.get('current_class')

                # 현재 상태를 기반으로 새로운 상태 결정
                new_class = '0' if current_class == '1' else '1'
                new_effnet_class = 'GOOD' if new_class == '1' else 'BAD'

                query = """
                    UPDATE classified_objects
                    SET
                        yolo_class = %s, effnet_class = %s,
                        is_reclassified = 1,
                        modified_at = NOW(), modified_by = %s
                    WHERE id = %s
                """
                cursor.execute(query, (new_class, new_effnet_class, modifier, item_id))
        conn.commit()
    except Exception as e:
        conn.rollback() # 오류 발생 시 모든 변경사항 되돌리기
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()

    return jsonify({'status': 'success'})

# --- 메모 업데이트 API ---
@app.route('/api/update_note', methods=['POST'])
@login_required # 👈 API도 보호합니다.
def update_note():
    data = request.json
    item_id = data.get('id')
    note = data.get('note')
    # 🔴 수정자를 현재 로그인된 사용자 이름으로 변경
    modifier = current_user.username

    if item_id is None: return jsonify({'status': 'error'}), 400
    
    query = """
        UPDATE classified_objects SET note = %s, modified_at = NOW(), modified_by = %s WHERE id = %s
    """
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(query, (note, modifier, item_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

@app.route('/stats/daily')
def stats_daily():
    end_date_str = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    start_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    start_date_str = request.args.get('start_date', (start_date - timedelta(days=30)).strftime('%Y-%m-%d'))
    model_gb = request.args.get('model_gb', 'S')
    query = """
        WITH RECURSIVE date_seq AS (
            SELECT %s AS dt
            UNION ALL
            SELECT DATE_ADD(dt, INTERVAL 1 DAY) FROM date_seq WHERE dt < %s
        )
        SELECT
            DATE_FORMAT(ds.dt, %s) AS std_date,
            COUNT(co.id) AS total_count,
            SUM(CASE WHEN co.yolo_class = '1' THEN 1 ELSE 0 END) AS good_count,
            SUM(CASE WHEN co.yolo_class = '0' THEN 1 ELSE 0 END) AS bad_count,
            ROUND(IFNULL(SUM(CASE WHEN co.yolo_class = '0' THEN 1 ELSE 0 END) / NULLIF(COUNT(co.id), 0) * 100, 0), 2) AS bad_rate
        FROM date_seq ds
        LEFT JOIN classified_objects co ON STR_TO_DATE(co.std_date, %s) = ds.dt AND co.del_yn = 'N' AND co.model_gb = %s
        GROUP BY ds.dt ORDER BY ds.dt
    """
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(query, (start_date_str, end_date_str, '%Y%m%d', '%Y%m%d', model_gb))
        result = cursor.fetchall()
    conn.close()
    return jsonify(result)

@app.route('/stats/weekly')
def stats_weekly():
    end_date_str = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    start_date_str = request.args.get('start_date', (end_date - timedelta(weeks=8)).strftime('%Y-%m-%d'))
    model_gb = request.args.get('model_gb', 'S')
    query = """
        SELECT
            YEARWEEK(STR_TO_DATE(std_date, %s), 1) AS year_week,
            COUNT(id) AS total_count,
            SUM(CASE WHEN yolo_class = '1' THEN 1 ELSE 0 END) AS ok_count,
            SUM(CASE WHEN yolo_class = '0' THEN 1 ELSE 0 END) AS ng_count,
            ROUND(IFNULL(SUM(CASE WHEN yolo_class = '0' THEN 1 ELSE 0 END) / COUNT(id) * 100, 0), 2) AS ng_rate
        FROM classified_objects
        WHERE del_yn = 'N' AND model_gb = %s AND STR_TO_DATE(std_date, %s) BETWEEN %s AND %s
        GROUP BY year_week
        ORDER BY year_week
    """
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(query, ('%Y%m%d', model_gb, '%Y%m%d', start_date_str, end_date_str))
        data = cursor.fetchall()
        # week_label을 Python에서 생성
        for i, row in enumerate(data):
            row['week_label'] = f"{- (len(data) - 1 - i)}주"
    conn.close()
    return jsonify(data)

@app.route('/stats/monthly')
def stats_monthly():
    end_date_str = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    start_date_str = request.args.get('start_date', (end_date - timedelta(days=180)).strftime('%Y-%m-%d'))
    model_gb = request.args.get('model_gb', 'S')

    query = """
        SELECT
            DATE_FORMAT(STR_TO_DATE(std_date, %s), %s) AS year_months,
            COUNT(id) AS total_count,
            SUM(CASE WHEN yolo_class = '1' THEN 1 ELSE 0 END) AS good_count,
            SUM(CASE WHEN yolo_class = '0' THEN 1 ELSE 0 END) AS bad_count,
            ROUND(IFNULL(SUM(CASE WHEN yolo_class = '0' THEN 1 ELSE 0 END) / COUNT(id) * 100, 0), 2) AS bad_rate
        FROM classified_objects
        WHERE del_yn = 'N' AND model_gb = %s AND STR_TO_DATE(std_date, %s) BETWEEN %s AND %s
        GROUP BY year_months
        ORDER BY year_months
    """
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(query, ('%Y%m%d', '%Y-%m', model_gb, '%Y%m%d', start_date_str, end_date_str))
        result = cursor.fetchall()
    conn.close()
    return jsonify(result)

@app.route('/stats/score_distribution')
def stats_score_distribution():
    model_gb = request.args.get('model_gb', 'S')

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT round(score, 3) AS score, COUNT(*) count FROM classified_objects
			WHERE del_yn = 'N' AND model_gb = %s
            GROUP BY round(score, 3)
            ORDER BY score
        """, (model_gb,))
        result = cursor.fetchall()
    conn.close()
    return jsonify(result)

@app.route('/stats/reclassification_trend')
def stats_reclassification_trend():
    model_gb = request.args.get('model_gb', 'S')

    conn = get_db_connection()
    with conn.cursor() as cursor:
        # 일간 재분류 횟수 (최근 7일)
        cursor.execute("""
            WITH RECURSIVE date_seq AS (
                SELECT DATE_SUB(CURDATE(), INTERVAL 7 DAY) AS dt
                UNION ALL
                SELECT DATE_ADD(dt, INTERVAL 1 DAY)
                FROM date_seq
                WHERE dt < CURDATE() -- DATE('2025-01-10')
            )
            SELECT
                DATE_FORMAT(ds.dt, %s) AS std_date,
                IFNULL(SUM(CASE WHEN IFNULL(co.is_reclassified, 0) AND DEL_YN != 'Y' THEN 1 ELSE 0 END), 0) AS re_count
            FROM
            date_seq ds
            LEFT JOIN classified_objects co
            ON DATE_FORMAT(co.modified_at, %s) = DATE_FORMAT(ds.dt, %s) AND co.model_gb = %s
            GROUP BY ds.dt
            ORDER BY ds.dt
        """, ('%Y%m%d', '%Y%m%d', '%Y%m%d', model_gb,))
        result = cursor.fetchall()
    conn.close()
    return jsonify(result)
    
@app.route('/stats/overall')
def stats_overall():
    model_gb = request.args.get('model_gb', 'S')

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(*) as total_count,
                   SUM(CASE WHEN yolo_class = '1' THEN 1 ELSE 0 END) as good_count,
                   SUM(CASE WHEN yolo_class = '0' THEN 1 ELSE 0 END) as bad_count
            FROM classified_objects WHERE del_yn = 'N' AND model_gb = %s
        """, (model_gb,))

        result = cursor.fetchone()
    conn.close()
    return jsonify(result)

# =================================================================
# 👇 아래의 라우트 함수들을 app.py의 `if __name__ == '__main__':` 라인 **앞에** 추가하세요.
# =================================================================

# --- 아이디 중복 확인 API ---
@app.route('/check_userid', methods=['POST'])
def check_userid():
    data = request.get_json()
    userid = data.get('userid')
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM users WHERE userid = %s", (userid,))
        existing_user = cursor.fetchone()
    conn.close()
    if existing_user:
        return jsonify({'available': False})
    else:
        return jsonify({'available': True})
    
# --- ✨ [새로 추가] 이메일 중복 확인 API ---
@app.route('/check_email', methods=['POST'])
def check_email():
    data = request.get_json()
    email = data.get('email')
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        existing_user = cursor.fetchone()
    conn.close()
    if existing_user:
        return jsonify({'available': False})
    else:
        return jsonify({'available': True})
    
# --- 관리자 코드 실시간 확인 API ---
@app.route('/check_admin_code', methods=['POST'])
def check_admin_code():
    data = request.get_json()
    admin_code = data.get('admin_code')
    # 설정된 비밀 코드와 일치하는지 확인
    if admin_code == app.config['ADMIN_SECRET_CODE']:
        return jsonify({'valid': True})
    else:
        return jsonify({'valid': False})

# --- 회원가입 라우트 (수정된 버전) ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    # 💡 [핵심 로직 1]
    # 사용자가 이미 로그인된 상태에서 이 페이지에 오려고 하면,
    # 이전 세션을 깨끗하게 로그아웃시켜서 충돌을 방지합니다.
    if current_user.is_authenticated:
        logout_user()

    if request.method == 'POST':
        # ... (POST 요청 처리 로직은 기존과 동일하므로 생략) ...
        userid = request.form['userid']
        password = request.form['password']
        password_confirm = request.form['password_confirm']
        name = request.form['name']
        email = request.form['email']
        company = request.form.get('company', '')
        role = request.form.get('role', '')
        terms = request.form.get('terms')
        admin_code = request.form.get('admin_code', '')
        is_admin_user = False

        if admin_code == app.config['ADMIN_SECRET_CODE']:
            is_admin_user = True

        if password != password_confirm:
            flash('비밀번호가 일치하지 않습니다.', 'error')
            return redirect(url_for('register'))
        if not terms:
            flash('이용약관에 동의해야 합니다.', 'error')
            return redirect(url_for('register'))

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE userid = %s OR email = %s", (userid, email))
            existing_user = cursor.fetchone()
            if existing_user:
                flash('이미 사용 중인 아이디 또는 이메일입니다.', 'error')
                conn.close()
                return redirect(url_for('register'))

            hashed_password = generate_password_hash(password)
            cursor.execute("""
                INSERT INTO users (userid, password_hash, name, email, company, role, is_admin)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (userid, hashed_password, name, email, company, role, is_admin_user))
        conn.commit()
        conn.close()

        flash('회원가입이 완료되었습니다. 로그인해주세요.', 'register_success')
        return redirect(url_for('login'))

    # GET 요청 시에는 회원가입 페이지만 보여줍니다.
    return render_template('register.html')

# --- 로그인 라우트 (수정) ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    # 💡 [핵심 로직 2]
    # 사용자가 이미 로그인된 상태라면, 로그인 페이지를 보여줄 필요 없이
    # 즉시 메인 페이지로 보냅니다.
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        # ... (POST 요청 처리 로직은 기존과 동일하므로 생략) ...
        userid = request.form['userid']
        password = request.form['password']

        if not userid or not password:
            flash('아이디와 비밀번호를 모두 입력해주세요.', 'login_error')
            return render_template('login.html', userid=userid)

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, userid, password_hash, name, email, company, role, is_admin FROM users WHERE userid = %s", (userid,))
            user_data = cursor.fetchone()
        conn.close()

        if user_data and check_password_hash(user_data['password_hash'], password):
            user = User(id=user_data['id'], userid=user_data['userid'], password_hash=user_data['password_hash'],
                        name=user_data['name'], email=user_data['email'],
                        company=user_data['company'], role=user_data['role'], is_admin=user_data['is_admin'])
            login_user(user)
            flash('로그인 되었습니다.', 'login_success')
            return redirect(url_for('index'))
        else:
            flash('아이디 또는 비밀번호가 올바르지 않습니다.', 'login_error')
            return render_template('login.html', userid=userid)

    # GET 요청 시에는 로그인 페이지만 보여줍니다.
    return render_template('login.html', userid='')

# --- 로그아웃 라우트 ---
@app.route('/logout')
@login_required # 로그아웃은 로그인된 사용자만 가능
def logout():
    logout_user()
    # 👇 [핵심 추가] 로그아웃 성공 시 특별 카테고리로 flash 메시지 추가
    flash('로그아웃 되었습니다.', 'logout_success')
    return redirect(url_for('index'))

# --- 현재 비밀번호 실시간 확인 API ---
@app.route('/check_current_password', methods=['POST'])
@login_required
def check_current_password():
    data = request.get_json()
    password = data.get('password')
    if current_user.check_password(password):
        return jsonify({'valid': True})
    else:
        return jsonify({'valid': False})

# --- 계정 관리 (회원정보 수정) 라우트 ---
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        # 👈 [핵심 수정 1] 사용자가 제출한 데이터를 딕셔너리로 저장
        form_data = {
            'name': request.form.get('name'),
            'email': request.form.get('email'),
            'company': request.form.get('company'),
            'role': request.form.get('role')
        }
        
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_new_password = request.form.get('confirm_new_password')

        conn = get_db_connection()
        # 이메일 중복 확인 (본인 제외)
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE email = %s AND id != %s", (form_data['email'], current_user.id))
            if cursor.fetchone():
                flash('이미 사용 중인 이메일입니다.', 'error')
                conn.close()
                # 👈 [핵심 수정 2] redirect 대신 render_template로 입력값 유지
                return render_template('profile.html', user_data=form_data)

        # 기본 정보 업데이트
        with conn.cursor() as cursor:
            cursor.execute("UPDATE users SET name = %s, email = %s, company = %s, role = %s WHERE id = %s",
                           (form_data['name'], form_data['email'], form_data['company'], form_data['role'], current_user.id))

        # 비밀번호 변경 로직
        if current_password:
            if not current_user.check_password(current_password):
                flash('현재 비밀번호가 일치하지 않습니다.', 'error')
                conn.close()
                return render_template('profile.html', user_data=form_data)
            
            if not (8 <= len(new_password) <= 16):
                flash('새 비밀번호는 8자 이상, 16자 이하로 설정해주세요.', 'error')
                conn.close()
                return render_template('profile.html', user_data=form_data)

            if new_password != confirm_new_password:
                flash('새 비밀번호가 일치하지 않습니다.', 'error')
                conn.close()
                return render_template('profile.html', user_data=form_data)
            
            new_password_hash = generate_password_hash(new_password)
            with conn.cursor() as cursor:
                cursor.execute("UPDATE users SET password_hash = %s WHERE id = %s", (new_password_hash, current_user.id))

        conn.commit()
        conn.close()
        # 👇 [핵심 수정] 'success' 카테고리를 'profile_success'로 변경합니다.
        flash('회원 정보가 성공적으로 수정되었습니다.', 'profile_success')
        return redirect(url_for('profile'))

    # GET 요청 시, user_data를 None 또는 빈 딕셔너리로 전달
    return render_template('profile.html', user_data={})

# --- 관리자 페이지: 회원 목록 ---
@app.route('/admin')
@admin_required # 관리자만 접근 가능
def admin_dashboard():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
        users = cursor.fetchall()
    conn.close()
    return render_template('admin.html', users=users)

# --- 🚀 [신규] 모델 관리 페이지 라우트 ---
@app.route('/admin/model')
@admin_required
def model_management():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        # 재학습에 반영될 이미지 개수 조회 (기존과 동일)
        cursor.execute("SELECT COUNT(*) as count FROM classified_objects WHERE is_reclassified = 1 AND del_yn = 'N'")
        reclassified_count = cursor.fetchone()['count']

        # 💡 [수정] 재학습 이력 목록을 페이지네이션으로 조회
        page = int(request.args.get('page', 1))
        per_page = 10 # 한 페이지에 10개씩 표시
        offset = (page - 1) * per_page
        
        # 날짜 필터링
        from_date = request.args.get('from_date')
        to_date = request.args.get('to_date')
        
        query_conditions = []
        params = []

        if from_date:
            query_conditions.append("DATE(created_at) >= %s")
            params.append(from_date)
        if to_date:
            query_conditions.append("DATE(created_at) <= %s")
            params.append(to_date)
        
        where_clause = "WHERE " + " AND ".join(query_conditions) if query_conditions else ""

        # 총 이력 개수 조회
        cursor.execute(f"SELECT COUNT(*) as total FROM retraining_jobs {where_clause}", tuple(params))
        total_jobs = cursor.fetchone()['total']

        # 현재 페이지의 이력 목록 조회
        params.extend([per_page, offset])
        cursor.execute(f"SELECT * FROM retraining_jobs {where_clause} ORDER BY id DESC LIMIT %s OFFSET %s", tuple(params))
        job_history = cursor.fetchall()

    conn.close()
    
    return render_template(
        'model_management.html', 
        reclassified_count=reclassified_count,
        job_history=job_history,
        total_jobs=total_jobs,
        page=page,
        per_page=per_page,
        from_date=from_date,
        to_date=to_date
    )

# --- 🚀 [신규] 재학습 이력 삭제 API ---
@app.route('/api/delete_job/<int:job_id>', methods=['POST'])
@admin_required
def delete_job(job_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 실행 중인 작업은 삭제하지 못하도록 방어
            cursor.execute("SELECT status FROM retraining_jobs WHERE id = %s", (job_id,))
            job = cursor.fetchone()
            if job and job['status'] in ['RUNNING', 'PENDING']:
                return jsonify({'status': 'error', 'message': '실행 중인 작업은 삭제할 수 없습니다.'}), 400
            
            # 작업 삭제
            result = cursor.execute("DELETE FROM retraining_jobs WHERE id = %s", (job_id,))
            conn.commit()
            
            if result > 0:
                return jsonify({'status': 'success', 'message': f'작업 ID {job_id} 이력이 삭제되었습니다.'})
            else:
                return jsonify({'status': 'error', 'message': '삭제할 작업을 찾지 못했습니다.'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if conn:
            conn.close()

# --- 🚀 [신규] 재분류된 이미지 개수만 알려주는 간단한 API ---
@app.route('/api/reclassified_count')
@login_required
def get_reclassified_count():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) as count FROM classified_objects WHERE is_reclassified = 1 AND del_yn = 'N'")
        count = cursor.fetchone()['count']
    conn.close()
    return jsonify({'count': count})

# --- ✨ [새로 추가] 관리자 권한 토글 API ---
@app.route('/admin/toggle_admin/<int:user_id>', methods=['POST'])
@admin_required
def toggle_admin(user_id):
    if user_id == current_user.id:
        flash('자기 자신의 권한은 변경할 수 없습니다.', 'error')
        return redirect(url_for('admin_dashboard'))

    conn = get_db_connection()
    with conn.cursor() as cursor:
        # ✨ [핵심 추가] 대상 사용자가 이미 관리자인지 확인
        cursor.execute("SELECT is_admin FROM users WHERE id = %s", (user_id,))
        target_user = cursor.fetchone()

        # 대상이 존재하고, 이미 관리자라면 변경을 막음
        if target_user and target_user['is_admin']:
            flash('다른 관리자의 권한은 변경할 수 없습니다.', 'error')
            conn.close()
            return redirect(url_for('admin_dashboard'))

        # 대상이 일반 사용자인 경우에만 권한을 관리자로 변경 (True로 고정)
        # NOT is_admin 대신 is_admin = TRUE를 사용
        cursor.execute("UPDATE users SET is_admin = TRUE WHERE id = %s", (user_id,))
    conn.commit()
    conn.close()
    flash(f'사용자(ID: {user_id})를 관리자로 임명했습니다.', 'success')
    return redirect(url_for('admin_dashboard'))

# --- [수정] 회원 삭제 라우트 ---
# 기존 로직은 거의 동일하지만, 다른 관리자를 삭제하지 못하도록 방어 로직 추가
@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    if user_id == current_user.id:
        flash('자기 자신을 삭제할 수 없습니다.', 'error')
        return redirect(url_for('admin_dashboard'))

    conn = get_db_connection()
    with conn.cursor() as cursor:
        # ✨ [핵심 추가] 삭제하려는 대상이 관리자인지 확인
        cursor.execute("SELECT is_admin FROM users WHERE id = %s", (user_id,))
        target_user = cursor.fetchone()
        if target_user and target_user['is_admin']:
            flash('다른 관리자 계정은 삭제할 수 없습니다.', 'error')
            conn.close()
            return redirect(url_for('admin_dashboard'))

        # 대상이 일반 사용자인 경우에만 삭제 실행
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
    conn.commit()
    conn.close()
    flash(f'사용자(ID: {user_id})가 삭제되었습니다.', 'success')
    return redirect(url_for('admin_dashboard'))

# --- 회원 탈퇴 처리 ---
@app.route('/delete_account', methods=['POST'])
@login_required
def delete_account():
    # 현재 로그인된 사용자의 ID를 가져옴
    user_id = current_user.id
    
    # 세션에서 로그아웃 처리
    logout_user()
    
    # 데이터베이스에서 사용자 정보 삭제
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
    conn.commit()
    conn.close()
    
    # 성공 메시지와 함께 메인 페이지로 리다이렉트
    flash('회원 탈퇴가 완료되었습니다. 이용해주셔서 감사합니다.', 'logout_success') # 로그아웃 성공과 동일한 카테고리 사용
    return redirect(url_for('index'))

# --- 🚀 [신규] 기간별 데이터 삭제 API ---
@app.route('/api/delete_by_date', methods=['POST'])
@admin_required
def delete_by_date():
    data = request.get_json()
    from_date = data.get('from_date')
    to_date = data.get('to_date')

    if not from_date or not to_date:
        return jsonify({'status': 'error', 'message': '기간을 선택해주세요.'}), 400

    query = "DELETE FROM classified_objects WHERE std_date BETWEEN %s AND %s"
    
    conn = get_db_connection()
    with conn.cursor() as cursor:
        deleted_count = cursor.execute(query, (from_date, to_date))
    conn.commit()
    conn.close()
    
    flash(f'{from_date}부터 {to_date}까지의 데이터 {deleted_count}건이 삭제되었습니다.', 'success')
    return jsonify({'status': 'success', 'deleted_count': deleted_count})

# --- 🚀 [신규] 전체 데이터 삭제 API ---
@app.route('/api/delete_all', methods=['POST'])
@admin_required
def delete_all():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        # TRUNCATE는 롤백이 불가능하지만, DELETE보다 훨씬 빠릅니다.
        cursor.execute("TRUNCATE TABLE classified_objects")
    conn.commit()
    conn.close()
    
    flash('모든 검사 데이터가 영구적으로 삭제되었습니다.', 'success')
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)