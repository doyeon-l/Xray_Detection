# ⭐ eventlet.monkey_patch()를 최상단에 위치시켜야 한다.
# import eventlet
# eventlet.monkey_patch()

import argparse
import shutil
import sys
from flask import Flask, render_template, jsonify, make_response, redirect, url_for, flash, request
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import pymysql
from datetime import datetime, time, timedelta
import csv
import io
import cv2
import uuid
import os
from dotenv import load_dotenv
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as nn
from PIL import Image
from ultralytics import YOLO
from pytorch_msssim import ms_ssim
from efficientnet_pytorch import EfficientNet
from model.models import EfficientNetClassifier, EfficientNetAutoencoder  # 직접 작성한 모델 클래스 import
from model.models import MSSSIMLoss
from functools import wraps
from flask import abort
import subprocess  # 재학습 스크립트 실행을 위한 import

# XAI (Grad-CAM) 라이브러리
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.grad_cam import GradCAM
import ttach as tta

import psutil # 프로세스 제어
from flask_socketio import SocketIO  # 웹 소켓

from flask_mailing import Mail, Message
import asyncio
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
from authlib.integrations.flask_client import OAuth
from flask import g
import albumentations as A

from train import IMAGES_PATH, LABELS_PATH, RETRAIN_DATASET_PATH
import requests

# 취소 요청을 저장할 전역 딕셔너리
# { '세션ID': True } 형태로 저장
CANCELLATION_REQUESTS = {}

# ===================================================================
# 1단계: 모든 확장 기능 객체를 먼저 생성합니다 (app 없이)
# ===================================================================
login_manager = LoginManager()
socketio = SocketIO()
mail = Mail()
oauth = OAuth()

# ===================================================================
# 2단계: Flask 앱을 생성하고 모든 설정을 여기에 집중시킵니다
# ===================================================================
# .env 파일 로드
load_dotenv()

# 디버깅: 환경 변수가 올바르게 로드되었는지 확인
print("--- 환경 변수 로드 확인 ---")
print(f"KAKAO_CLIENT_ID: {os.getenv('KAKAO_CLIENT_ID')}")
print(f"KAKAO_CLIENT_SECRET: {os.getenv('KAKAO_CLIENT_SECRET')}")
print("--------------------------")

app = Flask(__name__)

# --- 모든 설정을 app.config에 로드합니다 ---
app.secret_key = 'fubao123'
app.config['ADMIN_SECRET_CODE'] = os.getenv('ADMIN_SECRET_CODE', 'admin123').strip()

# 카카오 설정
app.config['KAKAO_CLIENT_ID'] = os.getenv('KAKAO_CLIENT_ID')
app.config['KAKAO_CLIENT_SECRET'] = os.getenv('KAKAO_CLIENT_SECRET')

# 구글 설정 (이름을 통일성 있게 변경 - 권장)
app.config['GOOGLE_CLIENT_ID'] = os.getenv('GOOGLE_CLIENT_ID')
app.config['GOOGLE_CLIENT_SECRET'] = os.getenv('GOOGLE_CLIENT_SECRET')

# Flask-Mailing 설정
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
app.config["MAIL_PORT"] = 587
app.config["MAIL_SERVER"] = 'smtp.gmail.com'
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USE_SSL"] = False
app.config["MAIL_FROM"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_FROM_NAME"] = 'FUBAO 알림'

# ===================================================================
# 3단계: 설정이 완료된 app 객체로 모든 확장 기능을 초기화합니다
# ===================================================================
login_manager.init_app(app)
login_manager.login_view = 'login'  # init_app 호출 후에 설정해야 합니다.

socketio.init_app(app)
mail.init_app(app)
oauth.init_app(app)

# ===================================================================
# 4단계: 초기화된 oauth 객체에 소셜 로그인을 등록합니다
# ===================================================================
# 이제 라이브러리가 app.config에서 자동으로 키를 찾아가므로 client_id 등을 직접 넘길 필요가 없습니다.
google = oauth.register(
    name='google',
    client_id=app.config['GOOGLE_CLIENT_ID'],
    client_secret=app.config['GOOGLE_CLIENT_SECRET'],
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

kakao = oauth.register(
    name='kakao',
    client_id=app.config['KAKAO_CLIENT_ID'],
    client_secret=app.config['KAKAO_CLIENT_SECRET'],
    api_base_url='https://kapi.kakao.com/',
    access_token_url='https://kauth.kakao.com/oauth/token',
    authorize_url='https://kauth.kakao.com/oauth/authorize',
    client_kwargs={'scope': 'profile_nickname profile_image account_email'},
    client_id_param_name='client_id',  # Authlib 이 Kakao 쪽에서 요구하는 방식대로 정확히 맞춰서 보내도록 보장한다.
    token_endpoint_auth_method='client_secret_post'  # 토큰 요청 시 반드시 POST
)

uploadPath = './static/upload'
modelPath = './model'
xaiResultPath = './static/xai_results'  # XAI 결과 저장 폴더

# XAI 결과 폴더가 없으면 생성
if not os.path.exists(xaiResultPath):
    os.makedirs(xaiResultPath)

s = URLSafeTimedSerializer(app.secret_key)

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

async def send_notification_email_async(subject, recipients, body):
    msg = Message(subject=subject, recipients=recipients, html=body)
    await mail.send_message(msg)

def send_notification_email(subject, recipients, body):
    asyncio.run(send_notification_email_async(subject, recipients, body))

def get_admin_emails():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT email FROM users WHERE is_admin = 1")
        admins = cursor.fetchall()
    conn.close()
    return [admin['email'] for admin in admins]

def log_audit_action(action, target_type=None, target_id=None, details=None):
    if not current_user.is_authenticated: return
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO audit_logs (user_id, user_name, action, target_type, target_id, details)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (current_user.id, current_user.name, action, target_type, target_id, details))
    conn.commit()
    conn.close()

class User(UserMixin):
    def __init__(self, id, userid, password_hash, name, email, company, role, is_admin, is_onboarding_complete, auth_provider='local', kakao_access_token=None):
        self.id = id
        self.username = userid
        self.password_hash = password_hash
        self.name = name
        self.email = email
        self.company = company
        self.role = role
        self.is_admin = is_admin
        self.is_onboarding_complete = is_onboarding_complete
        self.auth_provider = auth_provider
        self.kakao_access_token = kakao_access_token

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT id, userid, password_hash, name, email, company, role, is_admin, is_onboarding_complete, auth_provider, kakao_access_token FROM users WHERE id = %s", (user_id,))
        user_data = cursor.fetchone()
    conn.close()
    if user_data:
        return User(id=user_data['id'], userid=user_data['userid'], password_hash=user_data['password_hash'], 
            name=user_data['name'], email=user_data['email'], 
            company=user_data['company'], role=user_data['role'], is_admin=user_data['is_admin'],
            is_onboarding_complete=user_data['is_onboarding_complete'],
            auth_provider=user_data.get('auth_provider', 'local'),
            kakao_access_token=user_data.get('kakao_access_token'))
    return None

# 모델 로드
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

def get_transform(size=300):  # B3 기준 300
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
        score = probabilities[0, predicted_class_index].item()  # 예측된 클래스의 확률을 점수로 사용
    return "GOOD" if predicted_class_index == 1 else "BAD", score

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# --- 카카오 연동 및 메시지 전송 관련 함수들 ---
@app.route('/profile/kakao/connect')
@login_required
def connect_kakao_account():
    code = request.args.get('code')
    rest_api_key = app.config['KAKAO_CLIENT_ID']
    redirect_uri = url_for('connect_kakao_account', _external=True)

    token_headers = {'Content-type': 'application/x-www-form-urlencoded;charset=utf-8'}
    token_data = {
        'grant_type': 'authorization_code',
        'client_id': rest_api_key,
        'redirect_uri': redirect_uri,
        'code': code,
    }
    token_res = requests.post('https://kauth.kakao.com/oauth/token', headers=token_headers, data=token_data)
    token_json = token_res.json()

    # 디버깅을 위해 카카오 토큰 응답을 출력한다.
    print(f"Kakao Token Response: {token_json}")

    access_token = token_json.get("access_token")
    refresh_token = token_json.get("refresh_token")
    
    # 'expires_in' 값을 안전하게 가져옵니다.
    # 만약 'expires_in' 키가 없거나 값이 None이면 기본값 0을 사용한다.
    expires_in_raw = token_json.get("expires_in", 0) 

    # 'expires_in_raw'가 유효한 숫자인지 확인하고 int로 변환한다.
    # 만약 유효하지 않으면 0으로 처리한다.
    try:
        expires_in = int(expires_in_raw)
    except (ValueError, TypeError):
        expires_in = 0 # 숫자로 변환할 수 없는 경우 대체 값

    # 토큰 만료 시간 계산
    expiry_time = datetime.now() + timedelta(seconds=expires_in)

    # DB에 토큰 정보 저장
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("""
            UPDATE users SET kakao_access_token = %s, kakao_refresh_token = %s, kakao_token_expiry = %s
            WHERE id = %s
        """, (access_token, refresh_token, expiry_time, current_user.id))
    conn.commit()
    conn.close()

    flash('카카오톡 계정이 성공적으로 연동되었습니다.', 'success')
    return redirect(url_for('profile'))

def send_kakao_message_to_admins(message_text):
    """관리자들에게 카카오톡 메시지를 보내는 함수"""
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT id, kakao_access_token, kakao_refresh_token, kakao_token_expiry FROM users WHERE is_admin = 1 AND kakao_access_token IS NOT NULL")
        admins = cursor.fetchall()
    conn.close()

    for admin in admins:
        # (실제 구현 시 토큰 갱신 로직 필요)
        send_kakao_message(admin['kakao_access_token'], message_text)

def send_kakao_message(access_token, message_text):
    """특정 사용자에게 메시지를 보내는 내부 함수"""
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-type': 'application/x-www-form-urlencoded;charset=utf-8'
    }
    # 간단한 텍스트 템플릿 사용
    template = {
        "object_type": "text",
        "text": message_text,
        "link": { "web_url": url_for('list_page', _external=True) }
    }
    data = {'template_object': str(template).replace("'", "\"")} # JSON 형식으로 변환
    
    res = requests.post('https://kapi.kakao.com/v2/api/talk/memo/default/send', headers=headers, data=data)
    if res.status_code != 200:
        print(f"카카오톡 메시지 발송 실패: {res.json()}")

# 외부에서 카카오톡 알림을 요청하는 API
@app.route('/api/send_kakao_notification', methods=['POST'])
def api_send_kakao_notification():
    data = request.json
    message = data.get('message')
    if not message:
        return jsonify({'status': 'error', 'message': '메시지가 없습니다.'}), 400
    
    send_kakao_message_to_admins(message)
    return jsonify({'status': 'success'})

def _process_files_background_task(files_data, std_date, model_gb, sid):
    """ AI 분석을 수행하고, 취소 요청을 확인하며, 웹소켓으로 진행률을 전송하는 백그라운드 작업 """
    total_files = len(files_data)
    try:
        for i, file_info in enumerate(files_data):
            # 1. 매 작업 시작 전, 취소 요청이 있었는지 확인
            if CANCELLATION_REQUESTS.get(sid):
                break  # 취소 요청이 있으면 루프를 중단
            
            org_image_name = file_info['name']
            filepath = file_info['path']

            # 2. 현재 처리 중인 파일 정보와 진행률을 웹소켓으로 전송
            socketio.emit('upload_progress', {
                'status': 'processing', 'current': i + 1, 'total': total_files,
                'filename': org_image_name
            }, to=sid)
            
            # 3. AI 분석 및 DB 저장
            img_pil = Image.open(filepath).convert('RGB')
            initial_prediction, yolo_class, effnet_class = 'UNKNOWN', '0', 'UNKNOWN'
            score, anomaly_score = 0.0, None

            if model_gb == 'S':
                yolo_results = yolo_model.predict(source=filepath, verbose=False)
                num_detections = len(yolo_results[0].boxes)
                confidence_threshold = 0.5
                if num_detections > 0 and yolo_results[0].boxes.conf[0].item() > confidence_threshold:
                    initial_prediction = "BAD"
                    yolo_class = '0'
                    score = yolo_results[0].boxes.conf[0].item()
                else:
                    initial_prediction = "GOOD"
                    yolo_class = '1'
                    score = yolo_results[0].boxes.conf[0].item() if num_detections > 0 else 1.0
                effnet_class = initial_prediction
            elif model_gb == 'U':
                transform = get_transform(size=224)
                input_tensor = transform(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    reconstructed = autoencoder_model(input_tensor)
                    reconstruction_error = nn.mse_loss(reconstructed, input_tensor).item()
                anomaly_score = reconstruction_error
                threshold = 0.6
                initial_prediction = "GOOD" if anomaly_score < threshold else "BAD"
                yolo_class = '1' if initial_prediction == 'GOOD' else '0'

                # 이상 점수가 임계값을 넘으면 관리자에게 알림
                if anomaly_score >= threshold:
                    admin_emails = get_admin_emails()
                    if admin_emails:
                        email_body = f"""
                        <h3>비지도 학습 모델 이상 감지 알림</h3>
                        <p>새로 업로드된 이미지에서 높은 이상 점수가 감지되었습니다.</p>
                        <ul>
                            <li><b>원본 파일명:</b> {org_image_name}</li>
                            <li><b>이상 점수:</b> {anomaly_score:.4f} (임계값: {threshold})</li>
                            <li><b>업로드 시간:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
                        </ul>
                        <p>시스템에 접속하여 확인해주세요.</p>
                        """
                        send_notification_email("[X-Ray 감지 시스템] 높은 이상 점수 감지", admin_emails, email_body)
                        kakao_message = f"🚨 높은 이상 점수 감지 🚨\n- 파일명: {org_image_name}\n- 이상 점수: {anomaly_score:.4f}"
                        send_kakao_message_to_admins(kakao_message)

                effnet_class = initial_prediction
                score = max(0.0, 1.0 - anomaly_score)

            conn = get_db_connection()
            with conn.cursor() as cursor:
                filename = os.path.basename(filepath)
                cursor.execute("""
                    INSERT INTO classified_objects
                        (std_date, model_gb, image_path, image_name, org_image_name, yolo_class, effnet_class, score, anomaly_score, initial_prediction)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (std_date, model_gb, filepath, filename, org_image_name, yolo_class, effnet_class, score, anomaly_score, initial_prediction))
                conn.commit()
            conn.close()

            socketio.sleep(0) # 다른 백그라운드 작업에 실행을 양보

        # for 루프가 완전히 끝난 후에, 여기서 단 한번만 최종 결과를 보낸다.
        if CANCELLATION_REQUESTS.get(sid):
            socketio.emit('upload_canceled', {'status': 'canceled', 'message': f'사용자에 의해 작업이 취소되었습니다.'}, to=sid)
        else:
            socketio.emit('upload_complete', {'status': 'success', 'message': f'{total_files}개 파일 분석 완료!'}, to=sid)

    except Exception as e:
        print(f"백그라운드 작업 중 오류 발생: {e}")
        socketio.emit('upload_complete', {'status': 'error', 'message': f'서버 처리 중 오류가 발생했습니다: {e}'}, to=sid)
    finally:
        if sid in CANCELLATION_REQUESTS:
            del CANCELLATION_REQUESTS[sid]

# 웹소켓을 통한 업로드 취소 이벤트 핸들러
@socketio.on('cancel_upload')
def handle_cancel_upload():
    # 이 이벤트를 보낸 클라이언트의 고유 ID (sid)를 키로 사용하여 취소 요청을 기록
    sid = request.sid
    print(f"클라이언트 [{sid}] 로부터 업로드 취소 요청을 받았습니다.")
    CANCELLATION_REQUESTS[sid] = True

@app.route('/upload', methods=['POST'])
@login_required
def upload_files():
    std_date = request.form.get('std_date')
    model_gb = request.form.get('model_gb', 'S')
    files = request.files.getlist('files')
    
    if not std_date or not files:
        return jsonify({'status': 'error', 'message': '기준일과 파일이 필요합니다.'}), 400

    # 1. 먼저 모든 파일을 서버에 저장하고, 처리할 파일 목록을 만든다.
    files_to_process = []
    for file in files:
        if file and allowed_file(file.filename):
            org_image_name = file.filename
            filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8] + '.jpg'
            filepath = os.path.join(uploadPath, filename)
            file.save(filepath)
            files_to_process.append({'name': org_image_name, 'path': filepath})

    # 2. 요청을 보낸 클라이언트의 고유 ID (sid)를 가져온다.
    sid = request.args.get('sid')  # sid를 flask_request에서 가져오도록 변경
    if not sid:
        return jsonify({'status': 'error', 'message': '웹소켓 세션 ID가 필요합니다.'}), 400

    # 3. 시간이 오래 걸리는 AI 분석 작업을 백그라운드에서 실행하도록 넘긴다.
    socketio.start_background_task(
        _process_files_background_task, 
        files_data=files_to_process, 
        std_date=std_date, 
        model_gb=model_gb, 
        sid=sid
    )
    
    # 4. "파일 수신은 끝났고, 이제부터 백그라운드 처리를 시작할게" 라는 의미로 즉시 응답한다.
    return jsonify({'status': 'processing_started'})

# XAI (Grad-CAM) 생성 API
@app.route('/api/grad_cam/<int:item_id>', methods=['GET'])
@login_required
def generate_grad_cam(item_id):
    conn = get_db_connection()
    with conn.cursor() as cursor:
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

        # 모델 구분에 따라 다른 XAI 로직을 실행
        if item['model_gb'] == 'S':
            # ----- 1. 지도학습 모델: Grad-CAM -----
            img_pil_resized = img_pil.resize((300, 300))
            rgb_img = np.array(img_pil_resized, dtype=np.float32) / 255
            transform = get_transform(size=300)
            input_tensor = transform(img_pil_resized).unsqueeze(0).to(device)

            target_layers = [classifier_model.features[-1]]
            cam = GradCAM(model=classifier_model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        elif item['model_gb'] == 'U':
            # --- 2. 비지도학습 모델: 복원 오차 맵 ---
            transform = get_transform(size=224)
            input_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                reconstructed_tensor = autoencoder_model(input_tensor)

            # 텐서를 시각화 가능한 이미지(numpy 배열)로 변환
            original_img_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            reconstructed_img_np = reconstructed_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            
            # 픽셀 단위로 차이를 계산 (오차 맵)
            error_map = np.abs(original_img_np - reconstructed_img_np)
            error_map_gray = np.mean(error_map, axis=2)  # 흑백으로 변환
            
            # 히트맵 생성
            heatmap = cv2.normalize(error_map_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # 원본 이미지도 0~255 범위의 uint8 타입으로 변환
            original_img_display = cv2.normalize(original_img_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # 원본 이미지와 히트맵을 합성
            superimposed_img = cv2.addWeighted(heatmap, 0.5, original_img_display, 0.5, 0)
            visualization = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)  # PIL 저장을 위해 RGB로 변환

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


# 성능 모니터링 API
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
        
        # fetchall()이 반환하는 튜플(tuple)을 리스트(list)로 변환
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
    
# 지도 학습 모델 재학습 트리거 API
@app.route('/admin/retrain_model', methods=['POST'])
@admin_required
def retrain_model():
    # 폼에서 하이퍼파라미터 값 받기
    epochs = request.form.get('epochs', 50, type=int)
    imgsz = request.form.get('imgsz', 224, type=int)
    # augmentation 파라미터 추가 (체크박스는 'on' 또는 None으로 값이 넘어옴)
    augment_flip = request.form.get('augment_flip') == 'on'
    augment_rotate = request.form.get('augment_rotate') == 'on'
    augment_contrast = request.form.get('augment_contrast') == 'on'

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 이미 실행 중인 작업이 있는지 확인
            cursor.execute("SELECT id FROM retraining_jobs WHERE status = 'RUNNING' OR status = 'PENDING'")
            if cursor.fetchone():
                flash('이미 재학습 작업이 진행 중입니다.', 'warning')
                return redirect(url_for('model_management'))

            # 작업 로그에 선택된 하이퍼파라미터 기록
            log_message = f"지도학습 재학습 작업을 대기열에 추가했습니다.\n"
            log_message += f" - Epochs: {epochs}, Image Size: {imgsz}\n"
            log_message += f" - Augmentations: Flip({augment_flip}), Rotate({augment_rotate}), Contrast({augment_contrast})\n\n"
            
            cursor.execute("INSERT INTO retraining_jobs (status, progress_log) VALUES ('PENDING', %s)", (log_message,))
            conn.commit()
            job_id = cursor.lastrowid

            # train.py에 인자 전달을 위한 command 리스트 구성
            command = [
                sys.executable, 'train.py', 
                '--job_id', str(job_id),
                '--epochs', str(epochs),
                '--imgsz', str(imgsz)
            ]
            # 체크박스가 선택된 경우에만 인자 추가
            if augment_flip: command.append('--augment_flip')
            if augment_rotate: command.append('--augment_rotate')
            if augment_contrast: command.append('--augment_contrast')
            
            process = subprocess.Popen(command)
            
            # 생성된 프로세스의 PID를 DB에 즉시 저장
            cursor.execute("UPDATE retraining_jobs SET process_id = %s WHERE id = %s", (process.pid, job_id))
            conn.commit()

        flash('모델 재학습 프로세스가 시작되었습니다.', 'success')
    except Exception as e:
        flash(f'재학습 시작 중 오류 발생: {e}', 'error')
    finally:
        if conn:
            conn.close()
    return redirect(url_for('model_management'))

# 비지도 학습 재학습 관련 API
@app.route('/api/unsupervised_retrain_count')
@admin_required
def get_unsupervised_retrain_count():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(*) as count FROM classified_objects 
            WHERE initial_prediction = 'BAD' AND yolo_class = '1' AND is_reclassified = 1 AND del_yn = 'N'
        """)
        count = cursor.fetchone()['count']
    conn.close()
    return jsonify({'count': count})

@app.route('/admin/retrain_autoencoder', methods=['POST'])
@admin_required
def retrain_autoencoder():
    # 폼에서 하이퍼파라미터 값 받기 (AJAX 요청이므로 request.form 사용)
    epochs = request.form.get('epochs', 10, type=int)
    learning_rate = request.form.get('lr', 0.0001, type=float)
    batch_size = request.form.get('batch_size', 16, type=int)

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM retraining_jobs WHERE status = 'RUNNING' OR status = 'PENDING'")
            if cursor.fetchone():
                return jsonify({'status': 'error', 'message': '이미 다른 재학습 작업이 진행 중입니다.'}), 409

            # 작업 로그에 선택된 하이퍼파라미터 기록
            log_message = f"비지도 학습 모델 재학습을 대기열에 추가했습니다.\n"
            log_message += f" - Epochs: {epochs}, Learning Rate: {learning_rate}, Batch Size: {batch_size}\n\n"

            cursor.execute("INSERT INTO retraining_jobs (status, progress_log) VALUES ('PENDING', %s)", (log_message,))
            conn.commit()
            job_id = cursor.lastrowid
            
            # train_autoencoder.py에 인자 전달
            command = [
                sys.executable, 'train_autoencoder.py',
                '--job_id', str(job_id),
                '--epochs', str(epochs),
                '--lr', str(learning_rate),
                '--batch_size', str(batch_size)
            ]

            process = subprocess.Popen(command)
            
            cursor.execute("UPDATE retraining_jobs SET process_id = %s WHERE id = %s", (process.pid, job_id))
            conn.commit()

        return jsonify({'status': 'success', 'message': '비지도 학습 모델 재학습 프로세스가 시작되었습니다.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if conn: conn.close()

# 재학습 중지 API
@app.route('/api/stop_retraining', methods=['POST'])
@admin_required
def stop_retraining_job():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 현재 실행중인 작업의 PID를 찾음.
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

# 재학습 상태 확인 API
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

# list.html 렌더링
@app.route('/list')
@login_required
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
        
    data_query = "SELECT id, std_date, model_gb, image_path, image_name, org_image_name, yolo_class, effnet_class, score, anomaly_score, initial_prediction, DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i:%%s') AS created_at, note, is_reclassified, modified_by, IFNULL(DATE_FORMAT(modified_at, '%%Y-%%m-%%d %%H:%%i:%%s'), '') AS modified_at, xai_image_path " + base_query + order_clause + " LIMIT %s OFFSET %s"
    
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
@login_required
def export_csv():
    # --- 1. 프론트엔드에서 보낸 모든 파라미터 받기 ---
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    model_gb = request.args.get('model_gb')
    yolo_class = request.args.get('yolo_class')
    is_reclassified = request.args.get('is_reclassified')
    
    # 포함할 컬럼 정보 받기
    include_score = request.args.get('include_score') == 'true'
    include_note = request.args.get('include_note') == 'true'
    include_history = request.args.get('include_history') == 'true'

    include_xai_path = request.args.get('include_xai_path') == 'true'  # XAI 경로 포함 여부

    # --- 2. 동적 쿼리 및 헤더 생성 ---
    select_clauses = [
        'id', 'std_date', 'org_image_name', 
        "IF(model_gb='S', '지도', '비지도') as model_type",
        "IF(yolo_class='1', 'GOOD', 'BAD') as status"
    ]
    headers = ['ID', '기준일', '원본 파일명', '모델', '판정']
    
    if include_score:
        # 지도학습의 'score'는 '신뢰도 점수'로, 비지도학습의 'anomaly_score'는 '이상 점수'로 명확히 분리
        select_clauses.append("CASE WHEN model_gb = 'S' THEN score ELSE NULL END as confidence_score")
        select_clauses.append("CASE WHEN model_gb = 'U' THEN anomaly_score ELSE NULL END as anomaly_score_val")
        headers.extend(['신뢰도 점수(지도)', '이상 점수(비지도)'])
    if include_note:
        select_clauses.append('note')
        headers.append('메모')
    if include_history:
        select_clauses.extend(['modified_by', "DATE_FORMAT(modified_at, '%%Y-%%m-%%d %%H:%%i:%%s') as modified_at_formatted"])
        headers.extend(['수정자', '수정일시'])
    if include_xai_path:
        # IFNULL 함수를 사용해 xai_image_path가 NULL이면 지정된 텍스트를, 아니면 원래 경로를 반환한다.
        select_clauses.append("IFNULL(xai_image_path, '생성되지 않음') as xai_path_status")
        headers.append('XAI 이미지 경로')

    # --- 3. 동적 WHERE 조건 생성 ---
    where_conditions = ["del_yn = 'N'"]
    params = []
    
    if from_date and to_date:
        where_conditions.append("std_date BETWEEN %s AND %s")
        params.extend([from_date, to_date])
    if model_gb in ('S', 'U'):
        where_conditions.append("model_gb = %s")
        params.append(model_gb)
    if yolo_class in ('0', '1'):
        where_conditions.append("yolo_class = %s")
        params.append(yolo_class)
    if is_reclassified in ('0', '1'):
        where_conditions.append("is_reclassified = %s")
        params.append(is_reclassified)
    
    # --- 4. 최종 쿼리 조합 및 실행 ---
    query = f"SELECT {', '.join(select_clauses)} FROM classified_objects WHERE {' AND '.join(where_conditions)} ORDER BY id DESC"
    
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(query, tuple(params))
        data = cursor.fetchall()
    conn.close()

    # --- 5. CSV 파일 생성 및 반환 (기존과 유사) ---
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    for row in data:
        writer.writerow(row.values())
    
    csv_data = output.getvalue().encode('utf-8-sig')
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = f"attachment; filename=export_{datetime.now().strftime('%Y%m%d%H%M')}.csv"
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
    # 감사 로그 기록
    log_audit_action('MOVE_TO_TRASH', details=f'{len(ids)} items')
    return jsonify({'status': 'success'})

# 재분류 API
@app.route('/api/reclassify', methods=['POST'])
@login_required
def api_reclassify():
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


# 선택 항목 일괄 재분류 API
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
        conn.rollback()  # 오류 발생 시 모든 변경사항 되돌리기
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()

    return jsonify({'status': 'success'})

# 기간별 일괄 판정 전환 API
@app.route('/api/reclassify_by_period', methods=['POST'])
@login_required
def api_reclassify_by_period():
    data = request.json
    from_date = data.get('from_date')
    to_date = data.get('to_date')
    from_class = data.get('from_class') # '0' or '1'
    to_class = data.get('to_class')     # '0' or '1'
    model_gb = data.get('model_gb')     # 'S' or 'U'
    
    if not all([from_date, to_date, from_class, to_class, model_gb]):
        return jsonify({'status': 'error', 'message': '모든 파라미터가 필요합니다.'}), 400

    modifier = current_user.username
    to_effnet_class = 'GOOD' if to_class == '1' else 'BAD'
    
    query = """
        UPDATE classified_objects
        SET yolo_class = %s, effnet_class = %s, is_reclassified = 1,
            modified_at = NOW(), modified_by = %s
        WHERE std_date BETWEEN %s AND %s
        AND model_gb = %s
        AND yolo_class = %s
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            updated_count = cursor.execute(query, (to_class, to_effnet_class, modifier, from_date, to_date, model_gb, from_class))
        conn.commit()
    except Exception as e:
        conn.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()

    return jsonify({'status': 'success', 'message': f'총 {updated_count}개 항목의 판정을 변경했습니다.'})

# 메모 업데이트 API
@app.route('/api/update_note', methods=['POST'])
@login_required
def update_note():
    data = request.json
    item_id = data.get('id')
    note = data.get('note')
    # 수정자를 현재 로그인된 사용자 이름으로 변경
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

# 아이디 중복 확인 API
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
    
# 이메일 중복 확인 API
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
    
# 관리자 코드 실시간 확인 API
@app.route('/check_admin_code', methods=['POST'])
def check_admin_code():
    # POST 요청이 아니거나, JSON이 아니면 에러 처리 (안정성 강화)
    if not request.is_json:
        return jsonify({"valid": False, "error": "Invalid request format"}), 400

    data = request.get_json()
    admin_code_from_user = data.get('admin_code')
    # .strip()을 추가하여 .env 파일의 잠재적인 공백 문제를 방지
    secret_code_from_config = app.config.get('ADMIN_SECRET_CODE', '').strip()

    # 입력값이 없거나, 설정값이 없는 경우를 처리
    if not admin_code_from_user or not secret_code_from_config:
        return jsonify({'valid': False})
    
    # .strip()으로 공백을 제거한 뒤 비교
    if admin_code_from_user.strip() == secret_code_from_config:
        return jsonify({'valid': True})
    else:
        return jsonify({'valid': False})

# 회원가입 라우트
@app.route('/register', methods=['GET', 'POST'])
def register():
    # 사용자가 이미 로그인된 상태에서 이 페이지에 오려고 하면,
    # 이전 세션을 깨끗하게 로그아웃시켜서 충돌을 방지한다.
    if current_user.is_authenticated:
        logout_user()

    if request.method == 'POST':
        userid = request.form['userid']
        password = request.form['password']
        password_confirm = request.form['password_confirm']
        name = request.form['name']
        email = request.form['email']
        company = request.form.get('company', '')
        role = request.form.get('role', '')
        terms = request.form.get('terms')

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
                INSERT INTO users (userid, password_hash, name, email, company, role, is_admin, is_onboarding_complete)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (userid, hashed_password, name, email, company, role, False, True))  # 자체 가입자는 is_onboarding_complete를 True로 간주
        conn.commit()

        # 가입 후 바로 로그인 시키고 온보딩 페이지로 이동 (또는 메인으로)
        # 이 부분은 정책에 따라 달라질 수 있다. 자체 가입 시에는 온보딩을 건너뛰고 바로 메인으로 보내도 좋다.
        # 여기서는 바로 로그인 시키고 메인으로 보내는 로직으로 수정한다.
        new_user_id = cursor.lastrowid
        user_obj = load_user(new_user_id)
        login_user(user_obj)
        conn.close()

        flash('회원가입이 완료되었습니다. 로그인해주세요.', 'register_success')
        return redirect(url_for('login'))

    # GET 요청 시에는 회원가입 페이지만 보여줌.
    return render_template('register.html')

# 로그인 라우트
@app.route('/login', methods=['GET', 'POST'])
def login():
    # 사용자가 이미 로그인된 상태라면, 로그인 페이지를 보여줄 필요 없이
    # 즉시 메인 페이지로 보낸다.
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        userid = request.form['userid']
        password = request.form['password']

        if not userid or not password:
            flash('아이디와 비밀번호를 모두 입력해주세요.', 'login_error')
            return render_template('login.html', userid=userid)

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, userid, password_hash, name, email, company, role, is_admin, is_onboarding_complete, auth_provider FROM users WHERE userid = %s", (userid,))
            user_data = cursor.fetchone()
        conn.close()

        if user_data and check_password_hash(user_data['password_hash'], password):
            user = User(id=user_data['id'], userid=user_data['userid'], password_hash=user_data['password_hash'],
                        name=user_data['name'], email=user_data['email'],
                        company=user_data['company'], role=user_data['role'], is_admin=user_data['is_admin'],
                        is_onboarding_complete=user_data['is_onboarding_complete'],
                        auth_provider=user_data.get('auth_provider', 'local'))
            login_user(user)
            flash('로그인 되었습니다.', 'login_success')
            return redirect(url_for('index'))
        else:
            flash('아이디 또는 비밀번호가 올바르지 않습니다.', 'login_error')
            return render_template('login.html', userid=userid)

    # GET 요청 시에는 로그인 페이지만 보여줌.
    return render_template('login.html', userid='')

# 로그아웃 라우트
@app.route('/logout')
@login_required # 로그아웃은 로그인된 사용자만 가능
def logout():
    logout_user()
    # 로그아웃 성공 시 특별 카테고리로 flash 메시지 추가
    flash('로그아웃 되었습니다.', 'logout_success')
    return redirect(url_for('index'))

# 현재 비밀번호 실시간 확인 API
@app.route('/check_current_password', methods=['POST'])
@login_required
def check_current_password():
    data = request.get_json()
    password = data.get('password')
    if current_user.check_password(password):
        return jsonify({'valid': True})
    else:
        return jsonify({'valid': False})

# 계정 관리 (회원정보 수정) 라우트
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        # 사용자가 제출한 데이터를 딕셔너리로 저장
        form_data = {
            'name': request.form.get('name'),
            'email': request.form.get('email'),
            'company': request.form.get('company'),
            'role': request.form.get('role')
        }

        # 관리자 코드 처리 로직
        admin_code_from_user = request.form.get('admin_code', '')
        
        # 관리자 코드가 입력되었고, 현재 유저가 일반 사용자인 경우에만 검증
        if admin_code_from_user and not current_user.is_admin:
            secret_code = app.config.get('ADMIN_SECRET_CODE', '').strip()
            
            if admin_code_from_user.strip() == secret_code:
                # 코드가 일치하면 is_admin 플래그를 True로 설정하여 업데이트
                conn = get_db_connection()
                with conn.cursor() as cursor:
                    cursor.execute("UPDATE users SET is_admin = TRUE WHERE id = %s", (current_user.id,))
                conn.commit()
                conn.close()
                flash('관리자 권한이 부여되었습니다! 다시 로그인하면 적용됩니다.', 'success')
            else:
                # 코드가 일치하지 않으면 오류 메시지 표시 후 현재 페이지 유지
                flash('입력하신 관리자 코드가 올바르지 않습니다.', 'error')
                return render_template('profile.html', user_data=form_data, auth_provider=current_user.auth_provider)
        
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
                # redirect 대신 render_template로 입력값 유지
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
        # 'success' 카테고리를 'profile_success'로 변경
        flash('회원 정보가 성공적으로 수정되었습니다.', 'profile_success')
        return redirect(url_for('profile'))

    # GET 요청 시, user_data를 None 또는 빈 딕셔너리로 전달
    return render_template('profile.html', user_data={}, auth_provider=current_user.auth_provider)

# 관리자 페이지: 회원 목록
@app.route('/admin')
@admin_required  # 관리자만 접근 가능
def admin_dashboard():
    # 필터링 로직 추가
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    is_admin = request.args.get('is_admin')

    query = "SELECT * FROM users"
    conditions = []
    params = []

    if from_date:
        conditions.append("DATE(created_at) >= %s")
        params.append(from_date)
    if to_date:
        conditions.append("DATE(created_at) <= %s")
        params.append(to_date)
    if is_admin in ('0', '1'):
        conditions.append("is_admin = %s")
        params.append(is_admin)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY created_at DESC"

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(query, tuple(params))
        users = cursor.fetchall()
    conn.close()
    return render_template('admin.html', users=users)
    # --- [수정 끝] ---

# 모델 관리 페이지 라우트
@app.route('/admin/model')
@admin_required
def model_management():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        # 재학습에 반영될 이미지 개수 조회 (기존과 동일)
        # 지도학습(S) 재학습에 반영될 이미지 개수만 정확히 조회
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM classified_objects 
            WHERE is_reclassified = 1 AND model_gb = 'S' AND del_yn = 'N'
        """)
        reclassified_count = cursor.fetchone()['count']

        # 재학습 이력 목록을 페이지네이션으로 조회
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

# 재학습 이력 삭제 API
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

# 재분류된 이미지 개수만 알려주는 간단한 API
@app.route('/api/reclassified_count')
@login_required
def get_reclassified_count():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) as count FROM classified_objects WHERE is_reclassified = 1 AND del_yn = 'N'")
        count = cursor.fetchone()['count']
    conn.close()
    return jsonify({'count': count})

# 관리자 권한 토글 API
@app.route('/admin/toggle_admin/<int:user_id>', methods=['POST'])
@admin_required
def toggle_admin(user_id):
    if user_id == current_user.id:
        flash('자기 자신의 권한은 변경할 수 없습니다.', 'error')
        return redirect(url_for('admin_dashboard'))

    conn = get_db_connection()
    with conn.cursor() as cursor:
        # 대상 사용자가 이미 관리자인지 확인
        cursor.execute("SELECT is_admin FROM users WHERE id = %s", (user_id,))
        target_user = cursor.fetchone()

        # 대상이 존재하고, 이미 관리자라면 변경을 막음.
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
    # 감사 로그 기록
    log_audit_action('PROMOTE_ADMIN', target_type='user', target_id=user_id)
    return redirect(url_for('admin_dashboard'))

# 회원 삭제 라우트
# 기존 로직은 거의 동일하지만, 다른 관리자를 삭제하지 못하도록 방어 로직 추가
@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    if user_id == current_user.id:
        flash('자기 자신을 삭제할 수 없습니다.', 'error')
        return redirect(url_for('admin_dashboard'))

    conn = get_db_connection()
    with conn.cursor() as cursor:
        # 삭제하려는 대상이 관리자인지 확인
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
    # 감사 로그 기록
    log_audit_action('DELETE_USER', target_type='user', target_id=user_id)
    return redirect(url_for('admin_dashboard'))

# 회원 탈퇴 처리
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
    flash('회원 탈퇴가 완료되었습니다. 이용해주셔서 감사합니다.', 'logout_success')  # 로그아웃 성공과 동일한 카테고리 사용
    return redirect(url_for('index'))

# ----- 휴지통 페이지 및 API -----
@app.route('/trash')
@login_required
def trash_page():
    # 이 페이지는 단순히 템플릿만 렌더링하고, 데이터는 API로 불러옴.
    return render_template('trash.html')

@app.route('/api/trash_list')
@login_required
def api_trash_list():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('limit', 20))
    offset = (page - 1) * per_page
    
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) as total FROM classified_objects WHERE del_yn = 'Y'")
        total_count = cursor.fetchone()['total']
        
        cursor.execute("""
            SELECT id, org_image_name, image_path, model_gb, yolo_class, DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i:%%s') AS created_at 
            FROM classified_objects WHERE del_yn = 'Y' 
            ORDER BY id DESC LIMIT %s OFFSET %s
        """, (per_page, offset))
        trash_items = cursor.fetchall()
    conn.close()
    
    return jsonify({'total': total_count, 'data': trash_items})

@app.route('/api/restore', methods=['POST'])
@login_required
def api_restore():
    ids = request.json.get('ids', [])
    if not ids: return jsonify({'status': 'error', 'message': '복원할 항목이 없습니다.'}), 400
    
    query = "UPDATE classified_objects SET del_yn = 'N' WHERE id IN (%s)" % ','.join(['%s'] * len(ids))
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(query, ids)
    conn.commit()
    conn.close()
    
    return jsonify({'status': 'success', 'message': f'{len(ids)}개 항목을 복원했습니다.'})

@app.route('/api/delete_permanent', methods=['POST'])
@login_required
def api_delete_permanent():
    ids = request.json.get('ids', [])
    if not ids: return jsonify({'status': 'error', 'message': '삭제할 항목이 없습니다.'}), 400
    
    # (주의) 실제 DELETE 쿼리
    query = "DELETE FROM classified_objects WHERE id IN (%s) AND del_yn = 'Y'" % ','.join(['%s'] * len(ids))
    conn = get_db_connection()
    with conn.cursor() as cursor:
        deleted_count = cursor.execute(query, ids)
    conn.commit()
    conn.close()

    return jsonify({'status': 'success', 'message': f'{deleted_count}개 항목을 영구 삭제했습니다.'})

@app.route('/api/empty_trash', methods=['POST'])
@login_required
def api_empty_trash():
    query = "DELETE FROM classified_objects WHERE del_yn = 'Y'"
    conn = get_db_connection()
    with conn.cursor() as cursor:
        deleted_count = cursor.execute(query)
    conn.commit()
    conn.close()

    return jsonify({'status': 'success', 'message': f'휴지통의 {deleted_count}개 항목을 모두 비웠습니다.'})

# 기간별 데이터 삭제 API
@app.route('/api/delete_by_date', methods=['POST'])
@admin_required
def delete_by_date():
    data = request.get_json()
    from_date = data.get('from_date')
    to_date = data.get('to_date')

    if not from_date or not to_date:
        return jsonify({'status': 'error', 'message': '기간을 선택해주세요.'}), 400

    # DELETE 대신, del_yn 플래그를 'Y'로 업데이트하는 쿼리로 변경
    query = "UPDATE classified_objects SET del_yn = 'Y' WHERE std_date BETWEEN %s AND %s"
    
    conn = get_db_connection()
    with conn.cursor() as cursor:
        deleted_count = cursor.execute(query, (from_date, to_date))
    conn.commit()
    conn.close()
    
    flash(f'{from_date}부터 {to_date}까지의 데이터 {deleted_count}건이 삭제되었습니다.', 'success')
    return jsonify({'status': 'success', 'deleted_count': deleted_count})

# 전체 데이터 삭제 API
@app.route('/api/delete_all', methods=['POST'])
@admin_required
def delete_all():
    # TRUNCATE 대신, 모든 항목의 del_yn 플래그를 'Y'로 업데이트하는 쿼리로 변경
    query = "UPDATE classified_objects SET del_yn = 'Y' WHERE del_yn = 'N'"

    conn = get_db_connection()
    deleted_count = 0
    try:
        with conn.cursor() as cursor:
            deleted_count = cursor.execute(query)
        conn.commit()
    except Exception as e:
        conn.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()
    
    return jsonify({'status': 'success', 'message': f'총 {deleted_count}개의 항목이 삭제 처리되었습니다.'})

# 계정 찾기 및 비밀번호 재설정
@app.route('/find_account', methods=['GET', 'POST'])
def find_account():
    if request.method == 'POST':
        email = request.form.get('email')
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT userid, auth_provider FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()
        conn.close()

        if user:
            # Google 계정인 경우 기능 차단
            if user['auth_provider'] == 'google':
                flash('Google 계정은 이 기능을 사용할 수 없습니다.', 'find_id_error')
                return redirect(url_for('find_account'))
            
            send_notification_email(
                "[X-Ray 감지 시스템] 아이디 찾기 결과",
                [email],
                f"<h3>요청하신 아이디는 <b>{user['userid']}</b> 입니다.</h3>"
            )
            flash('입력하신 이메일로 아이디 정보를 발송했습니다.', 'find_id_success')
        else:
            flash('해당 이메일로 가입된 계정을 찾을 수 없습니다.', 'find_id_error')
        return redirect(url_for('find_account'))
    return render_template('find_account.html')

@app.route('/reset_password_request', methods=['POST'])
def reset_password_request():
    email = request.form.get('email')
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT id, auth_provider FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
    conn.close()

    if user:
        # Google 계정인 경우 기능 차단
        if user['auth_provider'] == 'google':
            flash('Google 계정은 비밀번호 재설정을 지원하지 않습니다. Google을 통해 직접 변경해주세요.', 'reset_pw_error')
            return redirect(url_for('find_account'))
        
        token = s.dumps(email, salt='password-reset-salt')
        reset_url = url_for('reset_password_token', token=token, _external=True)
        send_notification_email(
            "[X-Ray 감지 시스템] 비밀번호 재설정 요청",
            [email],
            f"<h3>비밀번호를 재설정하려면 아래 링크를 클릭하세요 (10분 유효):</h3><a href='{reset_url}'>{reset_url}</a>"
        )
        flash('비밀번호 재설정 링크를 이메일로 발송했습니다.', 'reset_pw_success')
    else:
        flash('해당 이메일로 가입된 계정을 찾을 수 없습니다.', 'reset_pw_error')
    return redirect(url_for('find_account'))

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password_token(token):
    try:
        email = s.loads(token, salt='password-reset-salt', max_age=600) # 10분 유효
    except SignatureExpired:
        flash('비밀번호 재설정 링크가 만료되었습니다. 다시 요청해주세요.', 'error')
        return redirect(url_for('find_account'))
    except Exception:
        flash('잘못된 링크입니다.', 'error')
        return redirect(url_for('find_account'))

    if request.method == 'POST':
        password = request.form['password']
        password_confirm = request.form['password_confirm']
        if password != password_confirm:
            flash('비밀번호가 일치하지 않습니다.', 'error')
            return render_template('reset_password.html', token=token)
        
        # 새 비밀번호가 현재 비밀번호와 같은지 서버에서 최종 확인
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT password_hash FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()
            
            if user and check_password_hash(user['password_hash'], password):
                conn.close()
                flash('새 비밀번호는 현재 비밀번호와 다르게 설정해야 합니다.', 'error')
                return render_template('reset_password.html', token=token)
        
        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("UPDATE users SET password_hash = %s WHERE email = %s", (hashed_password, email))
        conn.commit()
        conn.close()
        flash('비밀번호가 성공적으로 재설정되었습니다. 새 비밀번호로 로그인하세요.', 'reset_pw_complete')
        return redirect(url_for('login'))

    return render_template('reset_password.html', token=token)

# 비밀번호 재설정 시, 새 비밀번호가 현재와 다른지 실시간으로 확인하는 API
@app.route('/api/check_new_password_is_different', methods=['POST'])
def check_new_password_is_different():
    data = request.get_json()
    token = data.get('token')
    new_password = data.get('new_password')

    if not token or not new_password:
        return jsonify({'is_different': False, 'message': '필수 정보 누락'}), 400

    try:
        # 토큰을 해독하여 이메일 정보를 얻음.
        email = s.loads(token, salt='password-reset-salt', max_age=600)
    except Exception:
        # 유효하지 않은 토큰
        return jsonify({'is_different': False, 'message': '유효하지 않은 요청입니다.'}), 401

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT password_hash FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
    conn.close()

    # 사용자가 존재하고, 새 비밀번호가 현재 비밀번호와 같다면
    if user and check_password_hash(user['password_hash'], new_password):
        return jsonify({'is_different': False})
    else:
        # 사용자가 없거나, 비밀번호가 다르면 '다른 비밀번호'로 간주하여 통과
        return jsonify({'is_different': True})

# Google 소셜 로그인
@app.route('/login/google')
def login_google():
    redirect_uri = url_for('authorize_google', _external=True)
    # prompt='select_account' 옵션을 추가하여 항상 계정 선택 화면을 띄운다.
    return google.authorize_redirect(redirect_uri, prompt='select_account')

@app.route('/login/google/callback')
def authorize_google():
    token = google.authorize_access_token()

    # userinfo 엔드포인트의 전체 URL을 직접 사용하여 사용자 정보를 가져온다.
    # 이 주소는 oauth.register 설정의 server_metadata_url 안에 정의되어 있다.
    user_info_response = google.get('https://openidconnect.googleapis.com/v1/userinfo')
    user_info = user_info_response.json()

    email = user_info['email']
    name = user_info['name']

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user_data = cursor.fetchone()
        
        if user_data:
            # 이미 가입된 사용자가 구글로 로그인 시, auth_provider를 'google'로 업데이트
            cursor.execute("UPDATE users SET auth_provider = 'google' WHERE id = %s", (user_data['id'],))
            conn.commit()
            user_obj = load_user(user_data['id'])
            login_user(user_obj)
            flash('Google 계정으로 로그인되었습니다.', 'login_success')
        else:
            # 임시 비밀번호를 랜덤 '문자열'로 생성
            # uuid.uuid4().hex는 'a1b2c3d4...' 형태의 32자리 랜덤 문자열을 생성한다.
            temp_password_string = uuid.uuid4().hex 
            temp_password_hash = generate_password_hash(temp_password_string)
            base_userid = email.split('@')[0]
            userid = base_userid
            counter = 1
            while True:
                cursor.execute("SELECT id FROM users WHERE userid = %s", (userid,))
                if not cursor.fetchone():
                    break
                userid = f"{base_userid}{counter}"
                counter += 1

            cursor.execute("""
                INSERT INTO users (userid, password_hash, name, email, is_admin, auth_provider)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (userid, temp_password_hash, name, email, False, 'google'))
            conn.commit()
            
            new_user_id = cursor.lastrowid
            user_obj = load_user(new_user_id)
            login_user(user_obj)
            # 온보딩 페이지로 리디렉션
            flash('가입이 완료되었습니다. 프로필을 완성해주세요.', 'success')
            return redirect(url_for('complete_profile'))
    conn.close()
    return redirect(url_for('index'))

@app.route('/login/kakao')
def login_kakao():
    """로그인 페이지에서 사용되는 소셜 로그인 시작 라우트입니다."""
    redirect_uri = url_for('authorize_kakao', _external=True)
    # prompt='login' 옵션으로 항상 카카오 로그인 창을 띄웁니다.
    return kakao.authorize_redirect(redirect_uri, prompt='login')

@app.route('/login/kakao/callback')
def authorize_kakao():
    """카카오 인증 후, 신규 가입 또는 로그인을 처리하는 콜백 함수입니다."""
    try:
        # client_id 값이 제대로 전달되고 있는지 확인
        print(">>> Kakao Callback - Using client_id:", app.config['KAKAO_CLIENT_ID'])

        # 토큰 요청 전 단계에서 로그 찍기
        token = kakao.authorize_access_token()
        # print(">>> Kakao Token Response:", token)
    except Exception as e:
        print(f"카카오 토큰 발급 오류: {e}")
        flash('카카오 인증 중 오류가 발생했습니다.', 'error')
        return redirect(url_for('login'))

    user_info_res = kakao.get('v2/user/me')
    user_info = user_info_res.json()
    # print(">>> Kakao User Info:", user_info)   # 유저 정보 로그 추가
    kakao_account = user_info.get('kakao_account')

    if not kakao_account or not kakao_account.get('email'):
        flash('카카오 계정에서 이메일 정보 제공에 동의해야 합니다.', 'error')
        return redirect(url_for('login'))
        
    email = kakao_account.get('email')

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user_data = cursor.fetchone()

        # DB에 해당 이메일 사용자가 이미 존재하면 -> 로그인 처리
        if user_data:
            user_obj = load_user(user_data['id'])
            login_user(user_obj)
            flash('카카오 계정으로 로그인되었습니다.', 'login_success')
            return redirect(url_for('index'))

        # DB에 해당 이메일 사용자가 없으면 -> 신규 가입 처리
        else:
            profile = kakao_account.get('profile')
            name = profile.get('nickname') if profile else "사용자"
            temp_password_hash = generate_password_hash(uuid.uuid4().hex)
            base_userid = email.split('@')[0].replace('.', '').replace('-', '')
            userid = base_userid
            counter = 1
            while True:
                cursor.execute("SELECT id FROM users WHERE userid = %s", (userid,))
                if not cursor.fetchone(): break
                userid = f"{base_userid}{counter}"
                counter += 1

            cursor.execute("""
                INSERT INTO users (userid, password_hash, name, email, is_admin, auth_provider)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (userid, temp_password_hash, name, email, False, 'kakao'))
            conn.commit()
            
            new_user_id = cursor.lastrowid
            user_obj = load_user(new_user_id)
            login_user(user_obj)
            return redirect(url_for('complete_profile'))
    conn.close()

@app.route('/complete_profile', methods=['GET', 'POST'])
@login_required
def complete_profile():
    # 이 페이지는 온보딩이 완료되지 않은 사용자만 접근 가능
    if current_user.is_onboarding_complete:
        return redirect(url_for('index'))

    if request.method == 'POST':
        company = request.form.get('company', '')
        role = request.form.get('role', '')
        admin_code_from_user = request.form.get('admin_code', '')
        
        # is_admin 상태를 결정할 변수
        is_admin_to_be = current_user.is_admin  # 기본값은 현재 상태 유지

        # 사용자가 관리자 코드를 '입력한 경우에만' 검증 로직을 실행
        if admin_code_from_user:
            secret_code = app.config.get('ADMIN_SECRET_CODE', '').strip()
            
            # .strip()으로 공백을 제거한 뒤 비교
            if admin_code_from_user.strip() == secret_code:
                is_admin_to_be = True # 코드가 맞으면 관리자로 설정
                flash('관리자 코드가 확인되었습니다. 관리자 권한이 부여됩니다.', 'success')
            else:
                # 코드를 입력했는데 틀렸을 경우, 여기서 함수를 중단하고 에러 메시지와 함께 페이지를 다시 보여줌
                flash('입력하신 관리자 코드가 올바르지 않습니다.', 'error')
                return render_template('complete_profile.html')

        # DB 업데이트 로직은 단 한 번만 실행
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE users SET company = %s, role = %s, is_admin = %s, is_onboarding_complete = TRUE WHERE id = %s",
                (company, role, is_admin_to_be, current_user.id)
            )
        conn.commit()
        conn.close()
        
        # DB 업데이트 후에는 항상 메인 페이지로 이동
        flash('프로필이 성공적으로 업데이트되었습니다!', 'profile_success')
        return redirect(url_for('index'))

    # GET 요청 시에는 그냥 페이지를 보여줌.
    return render_template('complete_profile.html')

@app.before_request
def check_onboarding():
    # 로그인 상태이고, 온보딩 페이지로 가는 중이 아니며, 온보딩을 아직 완료하지 않았다면
    if current_user.is_authenticated \
        and request.endpoint not in ['complete_profile', 'logout', 'static', 'check_admin_code'] \
        and not current_user.is_onboarding_complete:
            # 강제로 추가 정보 입력 페이지로 보냄
            return redirect(url_for('complete_profile'))


# 감사 로그
@app.route('/admin/audit_logs')
@admin_required
def audit_logs():
    page = int(request.args.get('page', 1))
    per_page = 20
    offset = (page - 1) * per_page

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) as total FROM audit_logs")
        total_logs = cursor.fetchone()['total']
        
        cursor.execute("SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT %s OFFSET %s", (per_page, offset))
        logs = cursor.fetchall()
    conn.close()

    return render_template('audit_logs.html', logs=logs, page=page, per_page=per_page, total_logs=total_logs)


# train.py에서 이메일 발송을 위한 API
@app.route('/api/admin_emails')
def api_get_admin_emails():
    return jsonify({'emails': get_admin_emails()})

@app.route('/api/send_email', methods=['POST'])
def api_send_email():
    data = request.json
    send_notification_email(data['subject'], data['recipients'], data['body'])
    return jsonify({'status': 'success'})

# train.py로부터 재학습 진행률을 받아 웹소켓으로 전송하는 API
@app.route('/api/update_retraining_progress', methods=['POST'])
def update_retraining_progress():
    data = request.json
    job_id = data.get('job_id')
    progress = data.get('progress')
    message = data.get('message')
    
    if job_id is not None and progress is not None:
        # 'retraining_progress' 라는 이름의 웹소켓 이벤트를 클라이언트로 전송
        socketio.emit('retraining_progress', {
            'job_id': job_id,
            'progress': progress,
            'message': message
        })
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Missing data'}), 400

def prepare_dataset(data, args):
    print("재학습용 데이터셋을 구성합니다...")
    if os.path.exists(RETRAIN_DATASET_PATH):
        shutil.rmtree(RETRAIN_DATASET_PATH)
    os.makedirs(IMAGES_PATH)
    os.makedirs(LABELS_PATH)

    # 전달받은 인자에 따라 동적으로 증강 파이프라인 구성
    transforms_list = []
    if args.augment_flip:
        transforms_list.append(A.HorizontalFlip(p=0.5))
    if args.augment_rotate:
        transforms_list.append(A.Rotate(limit=15, p=0.5))
    if args.augment_contrast:
        transforms_list.append(A.RandomBrightnessContrast(p=0.3))
    
    transform = A.Compose(transforms_list)

    for item in data:
        original_image_path = item['image_path']
        final_class = item['yolo_class']

        if not os.path.exists(original_image_path):
            print(f"경고: 이미지 파일을 찾을 수 없습니다 - {original_image_path}")
            continue

        image = cv2.imread(original_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 원본 이미지 저장 및 라벨 생성
        original_basename = os.path.basename(original_image_path)
        cv2.imwrite(os.path.join(IMAGES_PATH, original_basename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        label_file_name = os.path.splitext(original_basename)[0] + '.txt'
        with open(os.path.join(LABELS_PATH, label_file_name), 'w') as f:
            if final_class == '0': # BAD
                f.write("0 0.5 0.5 0.5 0.5\n")

        # 'BAD' 데이터이고 증강 옵션이 선택된 경우에만 증강 실행
        if final_class == '0' and transforms_list:
            for i in range(2): # 증강 이미지 2개 생성
                augmented = transform(image=image)
                augmented_image = augmented['image']
                
                aug_basename = f"aug_{i}_{original_basename}"
                cv2.imwrite(os.path.join(IMAGES_PATH, aug_basename), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

                aug_label_name = os.path.splitext(aug_basename)[0] + '.txt'
                with open(os.path.join(LABELS_PATH, aug_label_name), 'w') as f:
                    f.write("0 0.5 0.5 0.5 0.5\n")
    print("데이터셋 구성 완료.")

# 자동 라벨링 제안 API (Placeholder)
@app.route('/api/auto_label/<int:item_id>')
@login_required
def auto_label(item_id):
    # **실제 구현 시 필요한 로직 (개념)**
    # 1. DB에서 item_id에 해당하는 이미지 경로 조회
    # 2. 비지도학습(Autoencoder) 모델로 원본 이미지와 복원 이미지 생성
    # 3. 원본과 복원 이미지의 차이(오차 맵) 계산 (scikit-image, opencv)
    # 4. 오차 맵에서 임계값을 초과하는 영역을 찾고, 가장 큰 영역을 선택 (Contour detection)
    # 5. 해당 영역을 감싸는 바운딩 박스 좌표(x,y,w,h) 계산 (cv2.boundingRect)
    # 6. 계산된 좌표를 JSON으로 반환
    
    # 아래는 기능 시연을 위한 임시(placeholder) 데이터이다.
    suggested_box = {'x': 100, 'y': 120, 'width': 50, 'height': 60, 'label': 'anomaly'}
    return jsonify({'status': 'success', 'box': suggested_box})

# 모델 버전별 성능 조회 API
@app.route('/stats/model_performance_by_version')
@login_required
def model_performance_by_version():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT version, performance 
            FROM retraining_jobs 
            WHERE status = 'COMPLETED' AND version IS NOT NULL
            ORDER BY id ASC
        """)
        data = cursor.fetchall()
    conn.close()
    return jsonify(data)

if __name__ == '__main__':
    # app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000, debug=True)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)