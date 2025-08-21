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
from model.model import EfficientNetAutoencoder # 👈 직접 작성한 모델 클래스 import 필요
from model.classifier import EfficientNetClassifier
# from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from model.model import MSSSIMLoss
from functools import wraps
from flask import abort

app = Flask(__name__)
uploadPath = './static/upload'
modelPath = './model'

# 💡 세션과 flash 메시지를 사용하기 위한 시크릿 키 설정 (실제 서비스에서는 더 복잡한 키 사용)
app.secret_key = 'your-secret-key-for-fubao-project'

# ✨ [새로 추가] 관리자 생성을 위한 비밀 코드 (실제 서비스에서는 환경 변수 등으로 안전하게 관리)
app.config['ADMIN_SECRET_CODE'] = 'admin123'

# --- Flask-Login 설정 ---
login_manager = LoginManager()
login_manager.init_app(app)
# 💡 로그인이 필요한 페이지에 비로그인 유저가 접근 시, 'login' 라우트로 리다이렉트
login_manager.login_view = 'login'

# login_manager 설정 아래에 추가
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403) # 403 Forbidden 오류 발생
        return f(*args, **kwargs)
    return decorated_function

# --- 사용자 모델 정의 (UserMixin 상속) ---
class User(UserMixin):
    def __init__(self, id, userid, password_hash, name, email, company, role, is_admin):
        self.id = id
        self.username = userid # Flask-Login은 'username' 속성을 내부적으로 사용할 수 있으므로 userid를 username에 할당
        self.password_hash = password_hash
        self.name = name
        self.email = email
        self.company = company
        self.role = role
        self.is_admin = is_admin # ✨ is_admin 속성 초기화

    # werkzeug.security를 사용한 비밀번호 처리
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# --- 사용자 로더 함수 ---
@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    with conn.cursor() as cursor:
        # 💡 DB에서 가져오는 필드들을 추가합니다.
        cursor.execute("SELECT id, userid, password_hash, name, email, company, role, is_admin FROM users WHERE id = %s", (user_id,))
        user_data = cursor.fetchone()
    conn.close()
    if user_data:
        # 💡 User 객체 생성 시 모든 필드를 전달합니다.
        return User(id=user_data['id'], userid=user_data['userid'], password_hash=user_data['password_hash'], 
                    name=user_data['name'], email=user_data['email'], 
                    company=user_data['company'], role=user_data['role'], is_admin=user_data['is_admin'])
    return None


def predict_with_classifier(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1) # 예측 확률
        predicted_class_index = torch.argmax(probabilities).item()
        score = probabilities[0, 1].item() # GOOD 클래스(인덱스 1)의 확률을 점수로 사용

    # 인덱스 0은 BAD, 1은 GOOD이라고 가정
    if predicted_class_index == 1:
        return "GOOD", score
    else:
        return "BAD", score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = EfficientNetClassifier(num_classes=2).to(device)
state_dict = torch.load(os.path.join(modelPath, 'eff_from_yolo_infer.pth'), map_location=device)
classifier.load_state_dict(state_dict)
classifier.eval()  # 평가 모드

model = EfficientNetAutoencoder(model_version='b2', output_size=224).to(device)
model.load_state_dict(torch.load(os.path.join(modelPath, 'autoencoder_effnetb2_img224_batch16_epoch100_M80_SS20.pth'), map_location=device))
model.eval()
# YOLO 모델 로드 함수 정의
def load_model(model_path='model/best.pt'):
    model = YOLO(model_path)
    return model

# DB 연결 설정
def get_db_connection():
    return pymysql.connect(
        host='127.0.0.1',
        # user='root2',
        # password='root12345',
        user='root',
        password='root123',
        db='mysql',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )



# 이미지 전처리
def get_transform(model_gb: str):
    if model_gb == 'S':  # 지도학습 (B3)
        size = 300
    elif model_gb == 'U':  # 비지도학습 (B2)
        size = 224
    else:
        size = 224  # fallback
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])



# YOLO로 객체 크롭 함수
def crop_image_with_yolo(image_path, yolo_model):
    img = cv2.imread(image_path)
    results = yolo_model(img)
    cropped_images = []
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cropped = img[y1:y2, x1:x2]
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            cropped_images.append(cropped_pil)
    
    return cropped_images if cropped_images else [Image.open(image_path)]  # 크롭 실패 시 원본 이미지 반환


# 여기까지 함수 정의 끝

# 전역 변수로 모델 로드
efficientnet_classifier_model = classifier
efficientnet_autoencoder_model = model
yolo_model = load_model()


UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 업로드 파일 유효성 검사
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 지도,비지도 학습 모델 업로드 
@app.route('/upload', methods=['POST'])
def upload():
    global efficientnet_classifier_model_b3, efficientnet_autoencoder_model
    std_date = request.form.get('std_date')
    model_gb = request.form.get('model_gb', 'S')  # 기본값은 'S'
    files = request.files.getlist('files')
    results = []

    for file in files:
        if file and allowed_file(file.filename):
            org_image_name = file.filename
            filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8] + '.jpg'
            filepath = os.path.join(uploadPath, filename)
            file.save(filepath)

            # 원본 이미지 로드
            original_img = Image.open(filepath).convert('RGB')
            
            # YOLO 모델로 박스 감지
            yolo_results = yolo_model(filepath)
            
            # 최종 분류 결과와 점수 변수 초기화
            effnet_class = 'UNKNOWN'
            yolo_class = '0' # 기본값은 '0' (BAD)
            score = 0.0

            # 💡 수정: YOLO 결과의 클래스 이름을 직접 확인하여 yolo_class 및 effnet_class 결정
            yolo_final_class = None
            if hasattr(yolo_results[0], 'boxes') and hasattr(yolo_results[0].boxes, 'cls'):
                for cls_tensor in yolo_results[0].boxes.cls:
                    class_name = yolo_results[0].names[int(cls_tensor)]
                    if class_name == 'BAD':
                        yolo_final_class = 'BAD'
                        break  # 'BAD'가 감지되면 바로 종료
                    elif class_name == 'GOOD':
                        yolo_final_class = 'GOOD'
            
            # =====================
            # 지도학습 (EfficientNet-B3 분류)
            # =====================
            if model_gb == 'S':
                # YOLO가 GOOD 또는 BAD를 감지했으면 그 결과를 사용
                if yolo_final_class:
                    effnet_class = yolo_final_class
                    yolo_class = '1' if effnet_class == 'GOOD' else '0'
                else:
                    # YOLO가 아무것도 감지하지 못했으면 B3 모델 사용
                    transform = get_transform('S') 
                    input_img_tensor = transform(original_img).unsqueeze(0).to(device)
                    effnet_class_raw, score = predict_with_classifier(efficientnet_classifier_model_b3, input_img_tensor)
                    
                    # 💡 최종 예측 결과 반전 로직 추가
                    if effnet_class_raw == 'GOOD':
                        effnet_class = 'BAD'
                    else:
                        effnet_class = 'GOOD'
                        
                    yolo_class = '1' if effnet_class == 'GOOD' else '0'

                print(f"[지도학습] 파일: {filename}, YOLO 클래스(박스 유무): {yolo_class}, 최종 예측 클래스: {effnet_class}, 점수: {score}")

            # =====================
            # 비지도학습 (EfficientNet 오토인코더)
            # =====================
            elif model_gb == 'U':
                reconstruction_error = None # 💡 reconstruction_error 변수를 None으로 초기화
                if yolo_final_class:
                    effnet_class = yolo_final_class
                    yolo_class = '1' if effnet_class == 'GOOD' else '0'
                else:
                    transform = get_transform('U')
                    input_tensor = transform(original_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        reconstructed_tensor = efficientnet_autoencoder_model(input_tensor)
                        reconstruction_error = nn.mse_loss(reconstructed_tensor, input_tensor).item()
                    score = max(0.0, 1.0 - reconstruction_error)
                    RECONSTRUCTION_THRESHOLD = 0.05
                    effnet_class_raw = 'GOOD' if reconstruction_error < RECONSTRUCTION_THRESHOLD else 'BAD'
                    
                    # 💡 최종 예측 결과 반전 로직 추가
                    if effnet_class_raw == 'GOOD':
                        effnet_class = 'BAD'
                    else:
                        effnet_class = 'GOOD'
                        
                    yolo_class = '1' if effnet_class == 'GOOD' else '0'

                print(f"[비지도학습] 파일: {filename}, YOLO 클래스(박스 유무): {yolo_class}, 최종 예측 클래스: {effnet_class}, 점수: {score}, 재구성 오류: {reconstruction_error}")

            # =====================
            # 공통: 데이터베이스 저장 로직 (단 한번 실행)
            # =====================
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO classified_objects 
                        (std_date, model_gb, image_path, image_name, org_image_name, yolo_class, effnet_class, score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (std_date, model_gb, filepath, filename, org_image_name, yolo_class, effnet_class, score))
                conn.commit()
            conn.close()
            
            # 결과를 results 리스트에 추가 (클라이언트에 응답하기 위함)
            results.append({
                'filename': filename,
                'yolo_class': yolo_class,
                'effnet_class': effnet_class,
                'score': score
            })

    return jsonify({'status': 'success', 'results': results})

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
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    yolo_class = request.args.get('yolo_class')
    model_gb = request.args.get('model_gb')  # 지도학습/비지도학습 선택
    search_term = request.args.get('search_term')
    sort_by = request.args.get('sort_by', 'id')
    sort_order = request.args.get('sort_order', 'DESC')

    query = """
        SELECT id, std_date, image_path, image_name, org_image_name, yolo_class, 
               effnet_class, score, DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i:%%s') AS created_at,
               note, is_reclassified, modified_by,
               IFNULL(DATE_FORMAT(modified_at, '%%Y-%%m-%%d %%H:%%i:%%s'), '') AS modified_at
        FROM classified_objects 
        WHERE del_yn = 'N'
    """
    params = []

    if from_date:
        from_date_dt = datetime.strptime(from_date, "%Y%m%d").strftime("%Y-%m-%d 00:00:00")
        query += " AND created_at >= %s"
        params.append(from_date_dt)
    if to_date:
        to_date_dt = datetime.strptime(to_date, "%Y%m%d").strftime("%Y-%m-%d 23:59:59")
        query += " AND created_at <= %s"
        params.append(to_date_dt)
    if yolo_class in ('0', '1'):
        query += " AND yolo_class = %s"
        params.append(yolo_class)
    if model_gb in ('S', 'U'):
        query += " AND model_gb = %s"
        params.append(model_gb)
    if search_term:
        query += " AND org_image_name LIKE %s"
        params.append(f"%{search_term}%")

    allowed_sort_columns = ['id', 'std_date', 'org_image_name', 'yolo_class', 'created_at']
    if sort_by in allowed_sort_columns and sort_order.upper() in ['ASC', 'DESC']:
        query += f" ORDER BY {sort_by} {sort_order.upper()}"
    else:
        query += " ORDER BY id DESC"

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(query, params)
        result = cursor.fetchall()
    conn.close()

    return jsonify(result)

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
@login_required # 👈 API도 보호합니다.
def api_reclassify():
    data = request.json
    item_id = data.get('id')
    new_class = data.get('new_class')
    # 🔴 수정자를 하드코딩된 "admin" 대신, 현재 로그인된 사용자 이름으로 변경
    modifier = current_user.username 

    if not item_id or new_class not in ('0', '1'):
        return jsonify({'status': 'error', 'message': 'Invalid data'}), 400
    
    query = """
        UPDATE classified_objects 
        SET 
            yolo_class = %s, effnet_class = %s, 
            is_reclassified = CASE WHEN is_reclassified = 1 THEN 0 ELSE 1 END,
            modified_at = NOW(), modified_by = %s
        WHERE id = %s
    """
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(query, (new_class, new_class, modifier, item_id))
    conn.commit()
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
    if request.method == 'POST':
        # 폼에서 모든 데이터 가져오기
        userid = request.form['userid']
        password = request.form['password']
        password_confirm = request.form['password_confirm']
        name = request.form['name']
        email = request.form['email']
        company = request.form.get('company', '') # 선택 사항
        role = request.form.get('role', '')       # 선택 사항
        terms = request.form.get('terms') # 약관 동의 체크박스

        admin_code = request.form.get('admin_code', '') # ✨ 관리자 코드 가져오기
        is_admin_user = False # 기본값은 일반 사용자

        # ✨ 관리자 코드 확인 로직
        if admin_code == app.config['ADMIN_SECRET_CODE']:
            is_admin_user = True

        # --- 서버 사이드 유효성 검사 ---
        if password != password_confirm:
            flash('비밀번호가 일치하지 않습니다.', 'error')
            return redirect(url_for('register'))
        
        if not terms:
            flash('이용약관에 동의해야 합니다.', 'error')
            return redirect(url_for('register'))

        conn = get_db_connection()
        with conn.cursor() as cursor:
            # 아이디 또는 이메일이 이미 존재하는지 다시 한번 확인
            cursor.execute("SELECT * FROM users WHERE userid = %s OR email = %s", (userid, email))
            existing_user = cursor.fetchone()
            if existing_user:
                flash('이미 사용 중인 아이디 또는 이메일입니다.', 'error')
                conn.close()
                return redirect(url_for('register'))

            # 비밀번호 해싱 및 DB에 저장
            hashed_password = generate_password_hash(password)
            cursor.execute("""
                INSERT INTO users (userid, password_hash, name, email, company, role, is_admin) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (userid, hashed_password, name, email, company, role, is_admin_user))
        conn.commit()
        conn.close()
        
        # 👇 [핵심 수정] 기존 'success' 카테고리를 'register_success'로 변경
        flash('회원가입이 완료되었습니다. 로그인해주세요.', 'register_success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

# --- 로그인 라우트 (수정) ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        userid = request.form['userid']
        password = request.form['password']
        
        # 클라이언트에서 빈칸을 막지만, 만약을 위한 서버 측 방어 코드
        if not userid or not password:
            # 👇 [핵심 수정 1] 카테고리를 'error'에서 'login_error'로 변경
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
            # 👇 [핵심 수정 2] 카테고리를 'error'에서 'login_error'로 변경
            flash('아이디 또는 비밀번호가 올바르지 않습니다.', 'login_error')
            return render_template('login.html', userid=userid)
            
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

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)