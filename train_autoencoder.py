import pymysql
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datetime import datetime
import argparse
import sys
import traceback
import requests
from model.models import EfficientNetAutoencoder

# ----- 설정 -----
DB_HOST = '127.0.0.1'
DB_USER = 'root'
DB_PASSWORD = 'root123'
DB_NAME = 'mysql'
MODEL_PATH = './model/autoencoder_effnetb2_img224_batch16_epoch100_M80_SS20.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DB 연결 및 업데이트 함수 (train.py와 동일)
def update_job_status(job_id, status, message=None, log_append=None):
    conn = None
    try:
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME, charset='utf8mb4', autocommit=True)
        with conn.cursor() as cursor:
            if log_append:
                cursor.execute("UPDATE retraining_jobs SET progress_log = CONCAT(IFNULL(progress_log, ''), %s) WHERE id = %s", (log_append, job_id))
            else:
                cursor.execute("UPDATE retraining_jobs SET status = %s, result_message = %s, completed_at = NOW() WHERE id = %s", (status, message, job_id))
    except Exception as e:
        print(f"DB 업데이트 실패: {e}")
    finally:
        if conn:
            conn.close()

class DatabaseLogger:
    def __init__(self, job_id):
        self.terminal = sys.stdout
        self.job_id = job_id

    def write(self, message):
        self.terminal.write(message)
        update_job_status(self.job_id, 'RUNNING', log_append=message)

    def flush(self):
        pass

# ----- 데이터 로딩을 위한 PyTorch Dataset 클래스 -----
class RetrainDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"이미지 로딩 오류: {img_path} - {e}")
            return None

# ----- 재학습할 데이터를 찾는 로직 -----
def fetch_newly_confirmed_good_data():
    print("DB에서 비지도 학습용 '신규 정상' 데이터를 가져옵니다...")
    conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    with conn.cursor() as cursor:
        # AI의 초기 예측은 'BAD'였으나, 사용자가 'GOOD'으로 재분류한 데이터를 찾습니다.
        query = """
            SELECT id, image_path FROM classified_objects 
            WHERE initial_prediction = 'BAD' 
              AND yolo_class = '1' 
              AND is_reclassified = 1
              AND del_yn = 'N'
        """
        cursor.execute(query)
        data = cursor.fetchall()
    conn.close()
    print(f"총 {len(data)}개의 비지도 학습용 데이터를 찾았습니다.")
    
    # 파일이 실제로 존재하는지 확인
    valid_data = [item for item in data if os.path.exists(item['image_path'])]
    return valid_data

# ----- 재학습 후 데이터 상태 초기화 -----
def reset_data_flags(item_ids):
    if not item_ids: return
    print("재학습에 사용된 데이터의 상태를 초기화합니다...")
    conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME, charset='utf8mb4')
    with conn.cursor() as cursor:
        # 재사용되지 않도록 initial_prediction을 현재 상태와 동일하게 업데이트
        placeholders = ','.join(['%s'] * len(item_ids))
        query = f"UPDATE classified_objects SET initial_prediction = 'GOOD' WHERE id IN ({placeholders})"
        cursor.execute(query, item_ids)
    conn.commit()
    conn.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", required=True, type=int)
    args = parser.parse_args()
    job_id = args.job_id

    sys.stdout = DatabaseLogger(job_id)
    sys.stderr = sys.stdout

    try:
        update_job_status(job_id, 'RUNNING', log_append="비지도 학습 모델 재학습을 시작합니다...\n")
        
        data_to_retrain = fetch_newly_confirmed_good_data()
        if not data_to_retrain:
            print("재학습할 데이터가 없습니다.\n")
            update_job_status(job_id, 'COMPLETED', message="재학습할 데이터가 없어 작업을 종료합니다.")
            sys.exit()

        image_paths = [item['image_path'] for item in data_to_retrain]
        item_ids = [item['id'] for item in data_to_retrain]

        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        dataset = RetrainDataset(image_paths, transform=transform)
        
        def collate_fn(batch):
            batch = list(filter(lambda x: x is not None, batch))
            return torch.utils.data.dataloader.default_collate(batch) if batch else None
            
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
        
        model = EfficientNetAutoencoder(model_version='b2', output_size=224).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.train()
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        epochs = 10
        print(f"총 {epochs} 에폭으로 미세조정을 시작합니다...")
        for epoch in range(epochs):
            total_loss = 0
            for batch_data in dataloader:
                if batch_data is None: continue
                img = batch_data.to(DEVICE)
                optimizer.zero_grad()
                output = model(img)
                loss = criterion(output, img)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            progress = int(((epoch + 1) / epochs) * 100)
            print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}\n")
            
            requests.post('http://127.0.0.1:5000/api/update_retraining_progress', 
                        json={'job_id': job_id, 'progress': progress, 'message': f'Epoch {epoch+1}/{epochs} 완료'})

        torch.save(model.state_dict(), MODEL_PATH)
        print("\n훈련된 모델을 저장했습니다.")
        
        reset_data_flags(item_ids)
        
        update_job_status(job_id, 'COMPLETED', message=f"재학습 완료 (Final Loss: {avg_loss:.4f})")

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"\n오류 발생: {e}\n{error_details}")
        update_job_status(job_id, 'FAILED', message=f"오류 발생: {e}")