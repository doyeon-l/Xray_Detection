# Xray_Detection
MBC아카데미컴퓨터학원 YOLO 팀 프로젝트: 20250729 ~ 20250829
----------------------------------------------------------------------------------------------------
## 01_create_tables.sql
-- ================================================================= --
-- 파일명: 01_create_tables.sql
-- 설명: 프로젝트의 모든 테이블 스키마를 생성합니다.
--       기존 테이블이 있다면 삭제 후 새로 생성되므로 주의가 필요합니다.
-- ================================================================= --

-- 데이터베이스를 선택합니다. (없다면 먼저 생성: CREATE DATABASE mysql;)
USE mysql;

-- 기존 테이블이 있다면 안전하게 삭제
DROP TABLE IF EXISTS classified_objects;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS retraining_jobs;


-- ================================================================= --
-- 1. 사용자 계정 테이블 (users)
-- ================================================================= --
CREATE TABLE `users` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `userid` VARCHAR(80) UNIQUE NOT NULL COMMENT '로그인 시 사용하는 아이디',
    `password_hash` VARCHAR(255) NOT NULL COMMENT '해시(Hash) 처리된 비밀번호',
    `name` VARCHAR(100) NOT NULL COMMENT '사용자 이름',
    `email` VARCHAR(120) UNIQUE NOT NULL COMMENT '이메일 주소',
    `company` VARCHAR(100) NULL COMMENT '소속 기관/회사 (선택)',
    `role` VARCHAR(100) NULL COMMENT '직무 또는 역할 (선택)',
    `is_admin` BOOLEAN NOT NULL DEFAULT FALSE COMMENT '관리자 여부 (True/False)',
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '계정 생성일'
) COMMENT='사용자 정보 테이블';

-- 테이블 생성 확인
DESC users;


-- ================================================================= --
-- 2. X-Ray 이미지 분류 결과 테이블 (classified_objects)
-- ================================================================= --
CREATE TABLE `classified_objects` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `std_date` VARCHAR(8) NOT NULL COMMENT '검출 기준일 (YYYYMMDD)',
    `model_gb` VARCHAR(1) NOT NULL DEFAULT 'S' COMMENT '사용 모델 구분 (S: 지도학습, U: 비지도학습)',
    `image_path` VARCHAR(1000) NULL COMMENT '이미지 파일 저장 경로',
    `image_name` VARCHAR(255) NULL COMMENT '시스템이 생성한 이미지 파일명',
    `org_image_name` VARCHAR(255) NULL COMMENT '업로드된 원본 이미지 파일명',
    `yolo_class` VARCHAR(1) NULL COMMENT '모델이 분류한 클래스 (0: BAD, 1: GOOD)',
    `effnet_class` VARCHAR(10) NULL COMMENT 'EfficientNet이 분류한 클래스 (GOOD/BAD)',
    `score` FLOAT NULL COMMENT '분류 모델의 신뢰도 점수',
    `anomaly_score` FLOAT NULL COMMENT '비지도 학습 모델의 이상 점수 (재구성 오류)',
    `initial_prediction` VARCHAR(10) NULL COMMENT '사용자 재분류 전 모델의 초기 예측 결과',
    `xai_image_path` VARCHAR(1000) NULL COMMENT 'Grad-CAM 결과 이미지 경로',
    `del_yn` VARCHAR(1) NOT NULL DEFAULT 'N' COMMENT '삭제 여부 (Y/N)',
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '생성일시',
    `note` TEXT NULL COMMENT '관리자 메모',
    `is_reclassified` BOOLEAN NOT NULL DEFAULT FALSE COMMENT '재분류 여부',
    `modified_by` VARCHAR(80) NULL COMMENT '수정자 userid',
    `modified_at` DATETIME NULL COMMENT '수정일시',
    KEY `idx_std_date_model_gb` (`std_date`, `model_gb`)
) COMMENT='X-Ray 이미지 분류 결과 데이터 테이블';

-- 테이블 생성 확인
DESC classified_objects;

-- ================================================================= --
-- 3. 모델 재학습 작업 관리 테이블 (retraining_jobs)
-- ================================================================= --
CREATE TABLE IF NOT EXISTS `retraining_jobs` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `process_id` INT NULL COMMENT '백그라운드에서 실행되는 train.py의 프로세스 ID (PID)',
    `status` VARCHAR(20) NOT NULL DEFAULT 'PENDING' COMMENT '작업 상태 (PENDING, RUNNING, COMPLETED, FAILED)',
    `progress_log` TEXT NULL COMMENT 'train.py 스크립트의 실시간 출력 로그',
    `result_message` VARCHAR(255) NULL COMMENT '최종 결과 메시지',
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '작업 생성 시간',
    `completed_at` TIMESTAMP NULL COMMENT '작업 완료 시간'
) COMMENT='모델 재학습 작업의 상태와 로그를 관리하는 테이블';

-- 테이블 생성 확인
DESC retraining_jobs;
