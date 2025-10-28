"""
플랫폼 1: 실시간 품질 예측 모니터링 플랫폼 (main_app.py)

- V11: (DB 연동)
  1. CSV 대신 PostgreSQL (kamp_db)에서 데이터 실시간 폴링
  2. ML 예측 결과를 process_data 테이블에 UPDATE
  3. 불량 예측 시 alarm_log 테이블에 INSERT
"""
import sys
import os
import pandas as pd
import joblib
import time
from dotenv import load_dotenv

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QTableWidget, QTableWidgetItem, QComboBox, 
    QPushButton, QHeaderView, QMessageBox # <<< QMessageBox 추가
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, pyqtSlot
from PyQt5.QtGui import QFont, QColor

import pyqtgraph as pg

# --- [신규/확인] SQLAlchemy 임포트 ---
from sqlalchemy import create_engine, text, exc # <<< exc 추가 (오류 처리)
from sqlalchemy.orm import sessionmaker
# --- [신규/확인] ---

# --- 1. 환경 변수 및 DB 설정 로드 ---
print("환경 변수 로딩...")
load_dotenv() 

DATABASE_URL = os.getenv('DATABASE_URL') # DB URL 사용
MODEL_PATH = os.getenv('MODEL_PATH')     # 모델 경로 사용
# DATA_PATH는 더 이상 사용하지 않음

OPTIMAL_VALUES = {
    'EX1.Z1_PV': 210.12,
    'EX1.MELT_P_PV': 6.86,
    'EX1.MD_PV': 71.32
}
print(f"DB URL: {DATABASE_URL}")
print(f"모델 경로: {MODEL_PATH}")

# --- [신규/확인] SQLAlchemy 엔진/세션 설정 ---
if not DATABASE_URL or not MODEL_PATH:
    print("오류: .env 파일에 DATABASE_URL 또는 MODEL_PATH가 설정되지 않았습니다.")
    # QMessageBox는 QApplication 생성 전에 사용할 수 없으므로 print 후 종료
    sys.exit("Environment variables not set.") 

try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    # DB 연결 테스트 (선택 사항)
    with engine.connect() as connection:
        print("DB 연결 성공.")
except exc.SQLAlchemyError as e:
     print(f"DB 연결 오류: {e}")
     sys.exit("Database connection failed.")
# --- [신규/확인] ---


# --- 2. ML Core (백엔드 QThread) ---
class MlCoreWorker(QThread):
    newData = pyqtSignal(dict) # UI 업데이트용 신호 (이제 DB Row + prediction 포함)
    db_error = pyqtSignal(str) # DB 오류 신호 추가

    def __init__(self, model_path): # <<< DATA_PATH 제거
        super().__init__()
        self.model_path = model_path
        self.running = True
        self.model_loaded = False
        self.last_processed_id = 0 # 마지막으로 처리한 process_data id

        try:
            # 모델 로드 (변경 없음)
            pipeline_data = joblib.load(self.model_path)
            self.scaler = pipeline_data['scaler']
            self.model = pipeline_data['model']
            self.features_order = pipeline_data['features'] # 모델 학습에 사용된 피처 순서 중요!
            self.model_loaded = True
            print("ML Core Worker: 모델 로드 성공.")
        except FileNotFoundError:
            print(f"오류: {self.model_path}를 찾을 수 없습니다.")
            # self.db_error.emit(f"모델 파일({self.model_path})을 찾을 수 없습니다.") # 스레드 시작 전이라 emit 불가
            self.model_loaded = False # 로드 실패 플래그
        except Exception as e:
            print(f"모델 로딩 중 오류: {e}")
            # self.db_error.emit(f"모델 로딩 오류: {e}")
            self.model_loaded = False

    def run(self):
        if not self.running or not self.model_loaded:
            print("ML Core Worker: 모델이 로드되지 않아 시작할 수 없음.")
            self.db_error.emit("모델 로드 실패. 앱을 종료하고 확인하세요.") # 시작 실패 알림
            return
            
        print("ML Core Worker: Starting DB polling...")
        
        db = None # finally 블록에서 사용하기 위해 미리 선언
        try:
            db = SessionLocal() # 스레드마다 새 DB 세션 생성

            # 시작 시 마지막 처리 ID 가져오기
            last_id_result = db.execute(text(
                "SELECT id FROM process_data WHERE prediction IS NOT NULL ORDER BY id DESC LIMIT 1"
            )).scalar()
            self.last_processed_id = last_id_result if last_id_result else 0
            print(f"ML Core Worker: Resuming from last processed ID: {self.last_processed_id}")


            while self.running:
                try:
                    # DB에서 새 데이터 조회
                    # `ROW` 객체는 컬럼 이름으로 접근 가능 (e.g., row.EX1_Z1_PV)
                    # SQLAlchemy 2.x에서는 ._mapping 속성으로 dict 변환
                    new_rows_cursor = db.execute(text(f"""
                        SELECT * FROM process_data 
                        WHERE prediction IS NULL AND id > :last_id
                        ORDER BY id ASC 
                        LIMIT 100
                    """), {"last_id": self.last_processed_id})
                    
                    new_rows = new_rows_cursor.fetchall() # 모든 결과 가져오기
                    
                    if not new_rows:
                        # 새 데이터가 없으면 잠시 대기 (CPU 사용 줄이기)
                        QThread.msleep(500) # time.sleep 대신 QThread.msleep 사용 (더 PyQt 친화적)
                        continue

                    # print(f"ML Core Worker: Processing {len(new_rows)} new rows...") # 너무 자주 출력되므로 주석 처리

                    for row in new_rows:
                        if not self.running: break 

                        # --- ML 예측 수행 ---
                        prediction = -1 # 오류 시 -1
                        status = "ERROR"
                        try:
                            # 1. 규칙 기반 체크
                            if row._mapping['EX1.MD_TQ'] == 0: # SQLAlchemy 2.x 방식
                                prediction = 1
                                status = "BAD"
                            else:
                                # 2. ML 모델 예측
                                # 필요한 피처만 추출 (Decimal -> float 변환 포함)
                                current_features_dict = {
                                     col: float(row._mapping[col]) 
                                     for col in self.features_order 
                                     if col in row._mapping # DB에 해당 컬럼이 있는지 확인
                                }
                                # 모델 학습 시 사용한 모든 피처가 있는지 확인
                                if len(current_features_dict) != len(self.features_order):
                                     missing = set(self.features_order) - set(current_features_dict.keys())
                                     raise ValueError(f"DB Row에 필요한 피처 누락: {missing}")

                                current_features_df = pd.DataFrame([current_features_dict])
                                current_features_scaled = self.scaler.transform(current_features_df)
                                prediction = self.model.predict(current_features_scaled)[0]
                                status = "BAD" if prediction == 1 else "GOOD"

                        except ValueError as ve: # 데이터 변환/누락 오류
                             print(f"데이터 준비 오류 (ID: {row.id}): {ve}")
                             # prediction은 -1, status는 "ERROR" 유지
                        except Exception as e:
                             print(f"ML 예측 오류 (ID: {row.id}): {e}")
                             # prediction은 -1, status는 "ERROR" 유지
                        
                        # --- DB 업데이트 ---
                        try:
                        # (수정 후) prediction을 int()로 변환
                            db.execute(text("UPDATE process_data SET prediction = :pred WHERE id = :row_id"), 
                            {"pred": int(prediction), "row_id": row.id}) # <<< int() 추가!
                            
                            # 4. 불량 예측 시 alarm_log 테이블에 INSERT
                            if prediction == 1:
                                db.execute(text("""
                                    INSERT INTO alarm_log (process_data_id, alarm_timestamp, status) 
                                    VALUES (:pd_id, :ts, 'New')
                                """), {"pd_id": row.id, "ts": row.timestamp}) 

                            db.commit() # 예측 결과 및 알람 로그 저장

                            # --- UI 업데이트 신호 발생 ---
                            row_dict = dict(row._mapping) 
                            
                            # 필요한 계산값 추가 (편차) - 오류 발생 가능성 고려 float() 사용
                            try:
                                dev_z1 = float(row._mapping['EX1.Z1_PV']) - OPTIMAL_VALUES['EX1.Z1_PV']
                                dev_melt_p = float(row._mapping['EX1.MELT_P_PV']) - OPTIMAL_VALUES['EX1.MELT_P_PV']
                                dev_md = float(row._mapping['EX1.MD_PV']) - OPTIMAL_VALUES['EX1.MD_PV']
                            except: # 혹시 모를 None 값 등에 대비
                                dev_z1, dev_melt_p, dev_md = 0.0, 0.0, 0.0

                            data_packet = {
                                'db_row': row_dict, 
                                'prediction': prediction,
                                'status': status,
                                'deviations': {'Z1': dev_z1, 'MELT_P': dev_melt_p, 'MD': dev_md}
                            }
                            self.newData.emit(data_packet)

                        except exc.SQLAlchemyError as db_upd_e: # DB 업데이트 오류
                            db.rollback() 
                            print(f"DB 업데이트 오류 (ID: {row.id}): {db_upd_e}")
                            self.db_error.emit(f"DB 업데이트 실패 (ID: {row.id}): {db_upd_e}")
                        except Exception as e: # 기타 오류 (emit 등)
                            print(f"UI 신호 발생 중 오류 (ID: {row.id}): {e}")

                        
                        # 마지막 처리 ID 업데이트 (오류 발생 여부와 관계없이 진행)
                        self.last_processed_id = row.id

                # 루프 종료 후 DB 세션 정리 (선택적)
                # db.commit() # 루프 내에서 커밋하므로 불필요

                except exc.SQLAlchemyError as db_poll_e: # DB 폴링 오류
                    print(f"DB 폴링 오류: {db_poll_e}")
                    self.db_error.emit(f"DB 연결 오류 발생. 재시도 중... ({db_poll_e})")
                    db.rollback() # 오류 시 롤백
                    # 연결 재시도 등을 위해 잠시 대기
                    QThread.sleep(5) # 5초 대기
                except Exception as e: # 기타 예상치 못한 오류
                    print(f"Worker 루프 오류: {e}")
                    # 오류 발생 시 잠시 대기 후 계속 시도
                    QThread.sleep(1)


            # --- While 루프 정상/비정상 종료 후 ---
            print("ML Core Worker: Stopping DB polling...")

        except Exception as e: # DB 세션 생성 자체 실패 등 초기 오류
            self.db_error.emit(f"Worker 시작 오류: {e}")
            print(f"ML Core Worker Initialization Error: {e}")
        finally:
            if db:
                db.close() # DB 세션 닫기
            print("ML Core Worker: Thread finished.")

    def stop(self):
        self.running = False
        print("ML Core Worker: Stopping signal received...")

# --- 3. 플랫폼 1: 메인 UI (PyQt) ---
class RealtimeDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("실시간 품질 예측 모니터링 플랫폼 (Platform 1)")
        self.setGeometry(100, 100, 1600, 900)
        
        # UI 상태 변수 추가
        self.total_count = 0
        self.good_count = 0
        self.bad_count = 0
        
        self.time_points = []
        self.param_z1_data = []
        self.param_melt_p_data = []
        self.param_md_data = []
        self.max_graph_points = 200
        
        self.initUI()
        
        # [수정] Worker 초기화 시 DATA_PATH 제거
        self.worker = MlCoreWorker(MODEL_PATH) 
        self.worker.newData.connect(self.update_ui)
        self.worker.db_error.connect(self.show_db_error) # DB 오류 메시지 박스 연결
        # [신규] 스레드 종료 시그널 연결 (선택적)
        self.worker.finished.connect(self.on_worker_finished) 
        self.worker.start()

    # initUI, create_header, create_tl_panel, create_tr_panel, 
    # create_bl_panel, create_br_panel 함수는 V9/V10 버전과 동일 (수정 X)
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QGridLayout()
        
        header_layout = self.create_header()
        
        panel_tl = self.create_tl_panel() 
        panel_tr = self.create_tr_panel() 
        panel_bl = self.create_bl_panel() 
        panel_br = self.create_br_panel() 

        main_layout.addLayout(header_layout, 0, 0, 1, 2)
        main_layout.addWidget(panel_tl, 1, 0)
        main_layout.addWidget(panel_tr, 1, 1) 
        main_layout.addWidget(panel_bl, 2, 0)
        main_layout.addWidget(panel_br, 2, 1) 
        
        main_layout.setRowStretch(1, 3); main_layout.setRowStretch(2, 3)
        main_layout.setColumnStretch(0, 3); main_layout.setColumnStretch(1, 2)
        
        central_widget.setLayout(main_layout)

    def create_header(self):
        # (V9/V10과 동일)
        header_layout = QHBoxLayout()
        brand_label = QLabel("KAMP - 소성가공 품질보증 AI 플랫폼")
        brand_label.setFont(QFont('Arial', 20, QFont.Bold))
        brand_label.setStyleSheet("color: #005A9C;")
        header_layout.addWidget(brand_label, 1) 
        
        model_selection_layout = QHBoxLayout()
        model_selection_layout.addWidget(QLabel("ML Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Model_A_Rule_LogiReg", "Model_B (dummy)", "Model_C (dummy)"])
        model_selection_layout.addWidget(self.model_combo)
        
        self.apply_button = QPushButton("적용")
        self.apply_button.clicked.connect(self.reset_ui) # [수정] restart -> reset_ui
        model_selection_layout.addWidget(self.apply_button)
        header_layout.addLayout(model_selection_layout)
        return header_layout

    def create_tl_panel(self):
        # (V9/V10과 동일)
        pg.setConfigOptions(antialias=True, background='w', foreground='k')
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Sensor Value')
        self.plot_widget.setLabel('bottom', 'DB ID (Part No.)') # X축 레이블 변경
        self.plot_widget.setTitle("주요 파라미터 실시간 모니터링", size='14pt')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()

        self.line_z1 = self.plot_widget.plot(pen=pg.mkPen('#0072B2', width=2), name="EX1.Z1_PV")
        self.line_melt_p = self.plot_widget.plot(pen=pg.mkPen('#D55E00', width=2), name="EX1.MELT_P_PV")
        self.line_md = self.plot_widget.plot(pen=pg.mkPen('#009E73', width=2), name="EX1.MD_PV")
        
        opt_z1 = pg.InfiniteLine(pos=OPTIMAL_VALUES['EX1.Z1_PV'], angle=0, pen=pg.mkPen('#0072B2', width=1, style=Qt.DotLine))
        opt_melt_p = pg.InfiniteLine(pos=OPTIMAL_VALUES['EX1.MELT_P_PV'], angle=0, pen=pg.mkPen('#D55E00', width=1, style=Qt.DotLine))
        opt_md = pg.InfiniteLine(pos=OPTIMAL_VALUES['EX1.MD_PV'], angle=0, pen=pg.mkPen('#009E73', width=1, style=Qt.DotLine))
        
        self.plot_widget.addItem(opt_z1); self.plot_widget.addItem(opt_melt_p); self.plot_widget.addItem(opt_md)
        return self.plot_widget

    def create_tr_panel(self):
        # (V9/V10과 동일)
        widget = QWidget()
        layout = QVBoxLayout()
        
        top_layout = QGridLayout()
        top_layout.addWidget(QLabel("Part No.:"), 0, 0)
        top_layout.addWidget(QLabel("Yield Rate:"), 1, 0)
        
        self.label_part_no = QLabel("N/A") # 초기값 변경
        self.label_part_no.setFont(QFont('Arial', 18, QFont.Bold))
        self.label_yield_rate = QLabel("N/A") # 초기값 변경
        self.label_yield_rate.setFont(QFont('Arial', 18, QFont.Bold))
        
        top_layout.addWidget(self.label_part_no, 0, 1)
        top_layout.addWidget(self.label_yield_rate, 1, 1)
        
        self.label_status = QLabel("WAITING...")
        self.label_status.setFont(QFont('Arial', 60, QFont.Bold))
        self.label_status.setAlignment(Qt.AlignCenter)
        self.label_status.setStyleSheet("background-color: #DDDDDD; color: #555555; border-radius: 10px;")
        
        layout.addLayout(top_layout)
        layout.addWidget(self.label_status, 1)
        widget.setLayout(layout)
        return widget

    def create_bl_panel(self):
        # (V9/V10과 동일)
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("실시간 데이터 로그"))
        
        hbox = QHBoxLayout()
        
        all_log_widget = QWidget()
        all_log_layout = QVBoxLayout()
        all_log_layout.addWidget(QLabel("All Log"))
        
        self.all_log_table = QTableWidget()
        self.all_log_table.setColumnCount(5)
        self.all_log_table.setHorizontalHeaderLabels(["ID", "Z1 Δ", "P Δ", "MD Δ", "Status"]) # 컬럼명 변경
        self.all_log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch) 
        
        all_log_layout.addWidget(self.all_log_table)
        all_log_widget.setLayout(all_log_layout)
        
        bad_log_widget = QWidget()
        bad_log_layout = QVBoxLayout()
        bad_log_layout.addWidget(QLabel("🚨 BAD Log 🚨"))
        
        self.bad_log_table = QTableWidget()
        self.bad_log_table.setColumnCount(4)
        self.bad_log_table.setHorizontalHeaderLabels(["ID", "Z1 Δ", "P Δ", "MD Δ"]) # 컬럼명 변경
        self.bad_log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch) 
        
        bad_log_layout.addWidget(self.bad_log_table)
        bad_log_widget.setLayout(bad_log_layout)
        
        hbox.addWidget(all_log_widget)
        hbox.addWidget(bad_log_widget)
        
        layout.addLayout(hbox)
        widget.setLayout(layout)
        return widget

    def create_br_panel(self):
        # (V9/V10과 동일)
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.bar_plot_widget = pg.PlotWidget()
        self.bar_plot_widget.setTitle("양품/불량 집계")
        self.bar_plot_widget.getAxis('bottom').setTicks([[(0.5, 'GOOD'), (1.5, 'BAD')]])
        self.bar_plot_widget.setLabel('left', 'Count')
        
        self.bar_graph_good = pg.BarGraphItem(x=[0.5], height=[0], width=0.6, brush='#009E73')
        self.bar_graph_bad = pg.BarGraphItem(x=[1.5], height=[0], width=0.6, brush='#D55E00')
        self.bar_plot_widget.addItem(self.bar_graph_good)
        self.bar_plot_widget.addItem(self.bar_graph_bad)
        
        self.bar_plot_widget.setYRange(0, 50)
        self.bar_plot_widget.setXRange(0, 2)
        
        self.good_text = pg.TextItem(text="", color=(0, 0, 0), anchor=(0.5, 0))
        self.bad_text = pg.TextItem(text="", color=(0, 0, 0), anchor=(0.5, 0))
        
        self.bar_plot_widget.addItem(self.good_text)
        self.bar_plot_widget.addItem(self.bad_text)
        
        layout.addWidget(self.bar_plot_widget, 1)
        
        info_layout = QGridLayout()
        info_layout.addWidget(QLabel("Total Count:"), 0, 0)
        self.label_br_total = QLabel("0")
        info_layout.addWidget(self.label_br_total, 0, 1)
        
        info_layout.addWidget(QLabel("Yield Rate:"), 1, 0)
        self.label_br_yield = QLabel("0.00 %")
        info_layout.addWidget(self.label_br_yield, 1, 1)
        
        layout.addLayout(info_layout)
        widget.setLayout(layout)
        return widget

    # --- [핵심 수정] ---
    @pyqtSlot(dict) # 명시적으로 Slot 지정 (선택 사항)
    def update_ui(self, data):
        """DB Worker로부터 신호를 받으면 UI 업데이트 (카운트 로직 포함)"""
        
        db_row = data['db_row']
        status = data['status']
        deviations = data['deviations']
        row_id = db_row['id'] # DB의 primary key (part no.로 사용)
        
        # ML 예측 오류 시 처리 (-1 값)
        if data['prediction'] == -1:
             print(f"Skipping UI update for ID: {row_id} due to prediction error.")
             # 선택적: 오류 로그 테이블 등에 기록
             return # UI 업데이트 건너뛰기

        # --- UI 상태 변수 업데이트 ---
        self.total_count += 1
        if status == "GOOD":
            self.good_count += 1
        else:
            self.bad_count += 1
        yield_rate = (self.good_count / self.total_count) * 100 if self.total_count > 0 else 0

        # --- (TL) 꺾은선 그래프 ---
        self.time_points.append(row_id) # X축을 DB id로 사용
        try: # DB 값이 None일 경우 대비
             z1_val = float(db_row.get('EX1.Z1_PV', 0) or 0)
             p_val = float(db_row.get('EX1.MELT_P_PV', 0) or 0)
             md_val = float(db_row.get('EX1.MD_PV', 0) or 0)
        except (ValueError, TypeError):
             z1_val, p_val, md_val = 0.0, 0.0, 0.0 # 오류 시 0으로 대체

        self.param_z1_data.append(z1_val)
        self.param_melt_p_data.append(p_val)
        self.param_md_data.append(md_val)
        
        if len(self.time_points) > self.max_graph_points:
            self.time_points.pop(0); self.param_z1_data.pop(0); self.param_melt_p_data.pop(0); self.param_md_data.pop(0)
            
        self.line_z1.setData(self.time_points, self.param_z1_data)
        self.line_melt_p.setData(self.time_points, self.param_melt_p_data)
        self.line_md.setData(self.time_points, self.param_md_data)
        
        # --- (TR) 실시간 상태 ---
        self.label_part_no.setText(f"{row_id}") 
        self.label_yield_rate.setText(f"{yield_rate:.2f} %") 
        if status == "GOOD":
            self.label_status.setText("GOOD"); self.label_status.setStyleSheet("background-color: #C8E6C9; color: #2E7D32; border-radius: 10px;")
        else: # BAD 또는 ERROR
            self.label_status.setText(status); self.label_status.setStyleSheet("background-color: #FFCDD2; color: #C62828; border-radius: 10px;")

        # --- (BL) 2분할 테이블 ---
        self.all_log_table.insertRow(0)
        item_index_all = QTableWidgetItem(str(row_id))
        item_z1_all = QTableWidgetItem(f"{deviations['Z1']:.2f}")
        item_p_all = QTableWidgetItem(f"{deviations['MELT_P']:.2f}")
        item_md_all = QTableWidgetItem(f"{deviations['MD']:.2f}")
        item_status_all = QTableWidgetItem(status)
        
        self.all_log_table.setItem(0, 0, item_index_all)
        self.all_log_table.setItem(0, 1, item_z1_all)
        self.all_log_table.setItem(0, 2, item_p_all)
        self.all_log_table.setItem(0, 3, item_md_all)
        self.all_log_table.setItem(0, 4, item_status_all)

        if self.all_log_table.rowCount() > 50: self.all_log_table.removeRow(50)
            
        if status == "BAD":
            bg_color = QColor(255, 205, 210)
            item_index_all.setBackground(bg_color); item_z1_all.setBackground(bg_color); 
            item_p_all.setBackground(bg_color); item_md_all.setBackground(bg_color);
            item_status_all.setBackground(bg_color)
            
            self.bad_log_table.insertRow(0)
            item_index_bad = QTableWidgetItem(str(row_id))
            item_z1_bad = QTableWidgetItem(f"{deviations['Z1']:.2f}")
            item_p_bad = QTableWidgetItem(f"{deviations['MELT_P']:.2f}")
            item_md_bad = QTableWidgetItem(f"{deviations['MD']:.2f}")
            
            self.bad_log_table.setItem(0, 0, item_index_bad)
            self.bad_log_table.setItem(0, 1, item_z1_bad)
            self.bad_log_table.setItem(0, 2, item_p_bad)
            self.bad_log_table.setItem(0, 3, item_md_bad)
            
            item_index_bad.setBackground(bg_color); item_z1_bad.setBackground(bg_color);
            item_p_bad.setBackground(bg_color); item_md_bad.setBackground(bg_color);

            if self.bad_log_table.rowCount() > 50: self.bad_log_table.removeRow(50)
            
        # --- (BR) 막대 그래프 + 텍스트 업데이트 ---
        counts = [self.good_count, self.bad_count] 
        
        self.bar_graph_good.setOpts(height=[counts[0]]) 
        self.bar_graph_bad.setOpts(height=[counts[1]])  
        
        self.good_text.setText(str(counts[0]))
        self.good_text.setPos(0.5, counts[0])
        
        self.bad_text.setText(str(counts[1]))
        self.bad_text.setPos(1.5, counts[1])
        
        max_count = max(counts) if max(counts) > 50 else 50
        self.bar_plot_widget.setYRange(0, max_count * 1.2)
        
        self.label_br_total.setText(f"{self.total_count}") 
        self.label_br_yield.setText(f"{yield_rate:.2f} %")  
        
    # [수정] restart_simulation -> reset_ui
    def reset_ui(self): 
        print("UI 리셋 요청...")
        
        # UI 상태 변수 초기화
        self.total_count = 0; self.good_count = 0; self.bad_count = 0
        
        # 그래프 데이터 초기화
        self.time_points.clear(); self.param_z1_data.clear(); self.param_melt_p_data.clear(); self.param_md_data.clear()
        self.line_z1.setData([], []); self.line_melt_p.setData([], []); self.line_md.setData([], [])
        
        # 테이블 초기화
        self.all_log_table.setRowCount(0); self.bad_log_table.setRowCount(0)
        
        # 막대그래프 및 텍스트 초기화
        self.bar_graph_good.setOpts(height=[0]); self.bar_graph_bad.setOpts(height=[0])
        self.good_text.setText("0"); self.good_text.setPos(0.5, 0)
        self.bad_text.setText("0"); self.bad_text.setPos(1.5, 0)
        self.label_br_total.setText("0"); self.label_br_yield.setText("0.00 %")

        # TR 패널 초기화
        self.label_part_no.setText("N/A"); self.label_yield_rate.setText("N/A")
        self.label_status.setText("WAITING..."); self.label_status.setStyleSheet("background-color: #DDDDDD; color: #555555; border-radius: 10px;")

        # [신규] Worker에게 마지막 처리 ID 리셋 알림 (선택적)
        # self.worker.reset_last_id() # <- Worker에 reset_last_id 메서드 추가 필요

        # (참고) 실제 모델 변경 로직은 추후 추가 필요
        selected_model = self.model_combo.currentText()
        print(f"'{selected_model}' 모델을 사용하는 것으로 간주합니다. (Worker는 계속 실행 중)")

    # [신규] DB 오류 메시지 박스 표시 슬롯
    @pyqtSlot(str)
    def show_db_error(self, error_message):
        print(f"DB Error Signal Received: {error_message}") # 터미널에도 출력
        # QMessageBox.critical(self, "데이터베이스 오류", error_message) # 너무 자주 뜨면 방해될 수 있음
        self.label_status.setText("DB ERROR!"); 
        self.label_status.setStyleSheet("background-color: #FFEBEE; color: #B71C1C; border-radius: 10px;") # 진한 빨강

    # [신규] Worker 종료 시그널 처리 슬롯 (선택적)
    @pyqtSlot()
    def on_worker_finished(self):
        print("Worker thread finished.")
        # 필요시 UI 상태 변경 (예: '연결 끊김' 표시)

    def closeEvent(self, event):
        print("윈도우 종료. ML Core 스레드를 중지합니다.")
        # 스레드가 정상적으로 종료될 때까지 잠시 기다릴 수 있음 (선택적)
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            # self.worker.wait(1000) # 최대 1초 대기
        event.accept()

# --- 4. 애플리케이션 실행 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RealtimeDashboard()
    window.show()
    sys.exit(app.exec_())