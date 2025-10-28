"""
플랫폼 1: 실시간 품질 예측 모니터링 플랫폼 (main_app.py)

- V17.2: (최종 문법 오류 수정)
  1. acknowledge_alarm, update_ui 함수 내 SyntaxError 수정 (try...except, indentation)
  2. 가독성 유지 (기능은 V17과 동일)
"""
import sys
import os
import pandas as pd
import joblib
import time
from datetime import datetime
from dotenv import load_dotenv

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QTableWidget, QTableWidgetItem, QComboBox,
    QPushButton, QHeaderView, QMessageBox, QSpacerItem, QSizePolicy,
    QFrame
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, pyqtSlot, QTimer, QSize
from PyQt5.QtGui import QFont, QColor

import pyqtgraph as pg

from sqlalchemy import create_engine, text, exc
from sqlalchemy.orm import sessionmaker

# --- 1. 환경 변수 및 DB 설정 로드 ---
print("환경 변수 로딩...")
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
MODEL_PATH = os.getenv('MODEL_PATH')

OPTIMAL_VALUES = { 'EX1.Z1_PV': 210.12, 'EX1.MELT_P_PV': 6.86, 'EX1.MD_PV': 71.32 }
print(f"DB URL: {DATABASE_URL}")
print(f"모델 경로: {MODEL_PATH}")

if not DATABASE_URL or not MODEL_PATH:
    print("오류: .env 파일에 DATABASE_URL 또는 MODEL_PATH가 없습니다.")
    sys.exit("Environment variables not set.")

try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with engine.connect() as connection:
        print("DB 연결 성공.")
except exc.SQLAlchemyError as e:
    print(f"DB 연결 오류: {e}")
    sys.exit("Database connection failed.")


# --- 2. ML Core (백엔드 QThread) ---
class MlCoreWorker(QThread):
    newData = pyqtSignal(dict)
    db_error = pyqtSignal(str)
    newWorkOrderDetected = pyqtSignal(str)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.running = True
        self.model_loaded = False
        self.last_processed_id = 0
        self.current_work_order = None

        try:
            pipeline_data = joblib.load(self.model_path)
            self.scaler = pipeline_data['scaler']
            self.model = pipeline_data['model']
            self.features_order = pipeline_data['features']
            self.model_loaded = True
            print("ML Core Worker: 모델 로드 성공.")
        except Exception as e:
            print(f"모델 로딩 중 오류: {e}")
            self.model_loaded = False # 오류 발생 시 False로 설정

    def run(self):
        if not self.running or not self.model_loaded:
            self.db_error.emit("모델 로드 실패.")
            return

        print("ML Core Worker: Starting DB polling...")
        db = None # finally에서 사용 위해 선언

        try:
            db = SessionLocal() # DB 세션 시작

            # 마지막 처리 상태 복원
            last_record = db.execute(text(
                "SELECT id, work_order FROM process_data WHERE prediction IS NOT NULL ORDER BY id DESC LIMIT 1"
            )).first()

            if last_record:
                self.last_processed_id = last_record.id
                self.current_work_order = last_record.work_order
                print(f"ML Core Worker: Resuming from ID: {self.last_processed_id}, WO: {self.current_work_order}")
            else:
                self.last_processed_id = 0
                print(f"ML Core Worker: Starting from beginning.")

            # 메인 폴링 루프
            while self.running:
                try:
                    # 현재 공정 정보 조회 (Generator가 업데이트한 값)
                    control_info = db.execute(text("SELECT current_count, target_count FROM process_control WHERE id = 1")).first()
                    current_db_count = control_info.current_count if control_info else 0
                    target_db_count = control_info.target_count if control_info else 0

                    # 새로운 데이터 조회 (Prediction이 안된 것)
                    new_rows_cursor = db.execute(text(f"SELECT * FROM process_data WHERE prediction IS NULL AND id > :last_id ORDER BY id ASC LIMIT 100"), {"last_id": self.last_processed_id})
                    new_rows = new_rows_cursor.fetchall()

                    # 새 데이터 없으면 대기
                    if not new_rows:
                        QThread.msleep(500) # 0.5초 대기
                        continue

                    # 새로운 각 행(row) 처리
                    for row in new_rows:
                        if not self.running: break # 스레드 중지 확인

                        row_mapping = row._mapping # 컬럼 접근용
                        row_work_order = row_mapping.get('work_order')

                        # 공정(Work Order) 변경 감지
                        if self.current_work_order is None: # 앱 시작 후 첫 데이터
                            self.current_work_order = row_work_order
                            print(f"ML Core Worker: Starting WO: {self.current_work_order}")
                            self.newWorkOrderDetected.emit(self.current_work_order or "Unknown")
                        elif row_work_order != self.current_work_order: # 공정 이름 변경 감지
                            print(f"ML Core Worker: New WO detected! From {self.current_work_order} to {row_work_order}")
                            self.current_work_order = row_work_order
                            self.newWorkOrderDetected.emit(self.current_work_order or "Unknown") # UI 리셋 시그널
                            QThread.msleep(100) # UI가 리셋될 시간 확보

                        # 변수 초기화
                        prediction = -1
                        status = "ERROR"
                        alarm_id = None

                        # 예측 수행 (규칙 + ML)
                        try:
                            if row_mapping['EX1.MD_TQ'] == 0:
                                prediction = 1
                                status = "BAD"
                            else:
                                features_dict = { col: float(row_mapping[col]) for col in self.features_order if col in row_mapping }
                                if len(features_dict) != len(self.features_order):
                                    missing = set(self.features_order) - set(features_dict.keys())
                                    raise ValueError(f"피처 누락: {missing}") # 오류 발생시킴

                                features_df = pd.DataFrame([features_dict])
                                features_scaled = self.scaler.transform(features_df)
                                prediction = int(self.model.predict(features_scaled)[0])
                                status = "BAD" if prediction == 1 else "GOOD"
                        except Exception as e:
                            print(f"ML/Data Error (ID: {row.id}): {e}")
                            # prediction=-1, status="ERROR" 유지됨

                        # DB 업데이트 및 알람 생성
                        try:
                            # Prediction 업데이트
                            db.execute(text("UPDATE process_data SET prediction = :pred WHERE id = :row_id"),
                                       {"pred": prediction, "row_id": row.id})

                            # Alarm Log 생성 (불량 시)
                            if prediction == 1:
                                result = db.execute(text("INSERT INTO alarm_log (process_data_id, alarm_timestamp, status) VALUES (:pd_id, :ts, 'New') RETURNING id"),
                                                    {"pd_id": row.id, "ts": row.timestamp})
                                alarm_id = result.scalar_one_or_none()

                            # 변경사항 커밋
                            db.commit()

                            # UI 업데이트용 데이터 준비
                            row_dict = dict(row_mapping)
                            try: # 편차 계산 (오류 방지)
                                dev_z1 = float(row_mapping['EX1.Z1_PV']) - OPTIMAL_VALUES['EX1.Z1_PV']
                                dev_melt_p = float(row_mapping['EX1.MELT_P_PV']) - OPTIMAL_VALUES['EX1.MELT_P_PV']
                                dev_md = float(row_mapping['EX1.MD_PV']) - OPTIMAL_VALUES['EX1.MD_PV']
                            except:
                                dev_z1, dev_melt_p, dev_md = 0.0, 0.0, 0.0

                            data_packet = {
                                'db_row': row_dict,
                                'prediction': prediction,
                                'status': status,
                                'deviations': {'Z1': dev_z1, 'MELT_P': dev_melt_p, 'MD': dev_md},
                                'alarm_id': alarm_id,
                                'current_count': current_db_count, # 현재 공정 카운트 전달
                                'target_count': target_db_count   # 현재 공정 목표 전달
                            }
                            # UI 업데이트 시그널 발생
                            self.newData.emit(data_packet)

                        # DB 업데이트/알람 생성 중 오류 처리
                        except exc.SQLAlchemyError as db_upd_e:
                            db.rollback() # 롤백 필수!
                            print(f"DB Update Err (ID: {row.id}): {db_upd_e}")
                            self.db_error.emit(f"DB Update Fail (ID: {row.id}): {db_upd_e}")
                        # 시그널 발생 중 오류 처리
                        except Exception as e:
                            print(f"Emit Signal Err (ID: {row.id}): {e}")

                        # 마지막 처리 ID 갱신 (오류 여부와 관계없이)
                        self.last_processed_id = row.id

                # DB 폴링/처리 중 오류 처리
                except exc.SQLAlchemyError as db_poll_e:
                    print(f"DB Poll Err: {db_poll_e}")
                    self.db_error.emit(f"DB Conn Err. Retrying... ({db_poll_e})")
                    if db: db.rollback() # 롤백 시도
                    QThread.sleep(5) # 5초 후 재시도
                except Exception as e:
                    print(f"Worker Loop Err: {e}")
                    QThread.sleep(1) # 1초 후 재시도

            # While 루프 정상 종료
            print("ML Core Worker: Stopping DB polling...")

        # Worker 시작 자체에서 오류 발생 시
        except Exception as e:
            self.db_error.emit(f"Worker Start Err: {e}")
            print(f"ML Core Worker Init Err: {e}")
        finally:
            # DB 세션 종료
            if db:
                db.close()
            print("ML Core Worker: Thread finished.")

    def stop(self):
        # 스레드 종료 플래그 설정
        self.running = False
        print("ML Core Worker: Stopping signal received...")


# --- 3. 플랫폼 1: 메인 UI (PyQt) ---
class RealtimeDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("실시간 품질 예측 모니터링 플랫폼 (Platform 1)")
        self.setGeometry(100, 100, 1600, 900)

        # UI 상태 변수
        self.total_count = 0
        self.good_count = 0
        self.bad_count = 0
        self.current_target_count = 0

        # 그래프 데이터
        self.time_points = []
        self.param_z1_data = []
        self.param_melt_p_data = []
        self.param_md_data = []
        self.max_graph_points = 200

        # DB 상태 폴링 타이머
        self.control_status_timer = QTimer(self)
        self.control_status_timer.timeout.connect(self.poll_control_status)
        self.control_status_timer.start(1000)

        # UI 생성
        self.initUI()

        # Worker 스레드
        self.worker = MlCoreWorker(MODEL_PATH)
        self.worker.newData.connect(self.update_ui)
        self.worker.db_error.connect(self.show_db_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.newWorkOrderDetected.connect(self.handle_new_work_order)
        self.worker.start()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QGridLayout()
        main_layout.setSpacing(10) # 패널 간 간격

        header_layout = self.create_header()
        main_layout.addLayout(header_layout, 0, 0, 1, 2)

        panel_tl_widget = self.create_tl_panel()
        panel_tr_widget = self.create_tr_panel()
        panel_bl_widget = self.create_bl_panel()
        panel_br_widget = self.create_br_panel()

        frame_tl = QFrame(); frame_tl.setFrameShape(QFrame.StyledPanel); frame_tl.setLayout(QVBoxLayout()); frame_tl.layout().addWidget(panel_tl_widget); frame_tl.layout().setContentsMargins(5,5,5,5)
        frame_tr = QFrame(); frame_tr.setFrameShape(QFrame.StyledPanel); frame_tr.setLayout(QVBoxLayout()); frame_tr.layout().addWidget(panel_tr_widget); frame_tr.layout().setContentsMargins(5,5,5,5)
        frame_bl = QFrame(); frame_bl.setFrameShape(QFrame.StyledPanel); frame_bl.setLayout(QVBoxLayout()); frame_bl.layout().addWidget(panel_bl_widget); frame_bl.layout().setContentsMargins(5,5,5,5)
        frame_br = QFrame(); frame_br.setFrameShape(QFrame.StyledPanel); frame_br.setLayout(QVBoxLayout()); frame_br.layout().addWidget(panel_br_widget); frame_br.layout().setContentsMargins(5,5,5,5)

        main_layout.addWidget(frame_tl, 1, 0)
        main_layout.addWidget(frame_tr, 1, 1)
        main_layout.addWidget(frame_bl, 2, 0)
        main_layout.addWidget(frame_br, 2, 1)

        main_layout.setRowStretch(1, 1) # 비율 동일하게
        main_layout.setRowStretch(2, 1)
        main_layout.setColumnStretch(0, 2) # 좌: 2
        main_layout.setColumnStretch(1, 1) # 우: 1

        central_widget.setLayout(main_layout)
        self.poll_control_status() # 초기 상태 호출

    def create_header(self):
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(10, 5, 10, 5)

        # 좌측 그룹
        left_group_layout = QHBoxLayout()
        brand_label = QLabel("KAMP - 소성가공 품질보증 AI 플랫폼")
        brand_label.setFont(QFont('Arial', 20, QFont.Bold))
        brand_label.setStyleSheet("color: #005A9C;")
        left_group_layout.addWidget(brand_label)
        left_group_layout.addSpacing(20)
        wo_layout = QHBoxLayout()
        wo_layout.addWidget(QLabel("Current WO:"))
        self.current_wo_label = QLabel("N/A")
        self.current_wo_label.setFont(QFont('Arial', 12, QFont.Bold))
        wo_layout.addWidget(self.current_wo_label)
        left_group_layout.addLayout(wo_layout)
        header_layout.addLayout(left_group_layout, 2) # 비율 2

        # 중앙 신축 공간
        header_layout.addStretch(1)

        # 우측 그룹
        right_group_layout = QHBoxLayout()
        right_group_layout.setSpacing(10)
        # 제어 버튼 그룹
        control_frame = QFrame(); control_frame.setFrameShape(QFrame.StyledPanel)
        control_button_layout = QHBoxLayout(); control_button_layout.setContentsMargins(5,2,5,2)
        self.start_button = QPushButton("▶ Start"); self.start_button.clicked.connect(self.start_process_control)
        self.pause_resume_button = QPushButton("❚❚ Pause"); self.pause_resume_button.clicked.connect(self.pause_resume_process_control)
        self.stop_button = QPushButton("■ Stop"); self.stop_button.clicked.connect(self.stop_process_control)
        self.start_button.setEnabled(False); self.pause_resume_button.setEnabled(False); self.stop_button.setEnabled(False)
        control_button_layout.addWidget(self.start_button); control_button_layout.addWidget(self.pause_resume_button); control_button_layout.addWidget(self.stop_button)
        control_frame.setLayout(control_button_layout)
        right_group_layout.addWidget(control_frame)
        # 상태 LED
        self.status_led = QLabel(); self.status_led.setFixedSize(QSize(25, 25)); self.status_led.setStyleSheet("background-color: gray; border-radius: 12px; border: 1px solid black;")
        right_group_layout.addWidget(self.status_led)
        # 모델/리셋 그룹
        model_reset_layout = QHBoxLayout(); model_reset_layout.addWidget(QLabel(" ML Model:"))
        self.model_combo = QComboBox(); self.model_combo.addItems(["Model_A_Rule_LogiReg", "Model_B (dummy)", "Model_C (dummy)"])
        model_reset_layout.addWidget(self.model_combo)
        self.reset_button = QPushButton("UI 리셋"); self.reset_button.clicked.connect(self.reset_ui)
        model_reset_layout.addWidget(self.reset_button)
        right_group_layout.addLayout(model_reset_layout)
        header_layout.addLayout(right_group_layout, 1) # 비율 1

        return header_layout

    def create_tl_panel(self):
        pg.setConfigOptions(antialias=True, background='w', foreground='k')
        plot_widget = pg.PlotWidget()
        plot_widget.setLabel('left', 'Sensor Value'); plot_widget.setLabel('bottom', 'DB ID (Part No.)')
        plot_widget.setTitle("주요 파라미터 실시간 모니터링", size='14pt'); plot_widget.showGrid(x=True, y=True, alpha=0.3); plot_widget.addLegend()
        self.line_z1 = plot_widget.plot(pen=pg.mkPen('#0072B2', width=2), name="EX1.Z1_PV")
        self.line_melt_p = plot_widget.plot(pen=pg.mkPen('#D55E00', width=2), name="EX1.MELT_P_PV")
        self.line_md = plot_widget.plot(pen=pg.mkPen('#009E73', width=2), name="EX1.MD_PV")
        opt_z1 = pg.InfiniteLine(pos=OPTIMAL_VALUES['EX1.Z1_PV'], angle=0, pen=pg.mkPen('#0072B2', width=1, style=Qt.DotLine))
        opt_melt_p = pg.InfiniteLine(pos=OPTIMAL_VALUES['EX1.MELT_P_PV'], angle=0, pen=pg.mkPen('#D55E00', width=1, style=Qt.DotLine))
        opt_md = pg.InfiniteLine(pos=OPTIMAL_VALUES['EX1.MD_PV'], angle=0, pen=pg.mkPen('#009E73', width=1, style=Qt.DotLine))
        plot_widget.addItem(opt_z1); plot_widget.addItem(opt_melt_p); plot_widget.addItem(opt_md)
        return plot_widget

    def create_tr_panel(self):
        widget = QWidget(); layout = QVBoxLayout(); layout.setContentsMargins(10, 10, 10, 10)
        top_layout = QGridLayout(); top_layout.addWidget(QLabel("진행 현황:"), 0, 0); top_layout.addWidget(QLabel("Yield Rate:"), 1, 0)
        self.label_part_no = QLabel("0 / 0"); self.label_part_no.setFont(QFont('Arial', 18, QFont.Bold))
        self.label_yield_rate = QLabel("N/A"); self.label_yield_rate.setFont(QFont('Arial', 18, QFont.Bold))
        top_layout.addWidget(self.label_part_no, 0, 1); top_layout.addWidget(self.label_yield_rate, 1, 1)
        self.label_status = QLabel("WAITING..."); self.label_status.setFont(QFont('Arial', 60, QFont.Bold)); self.label_status.setAlignment(Qt.AlignCenter); self.label_status.setStyleSheet("background-color: #DDDDDD; color: #555555; border-radius: 10px; padding: 10px;")
        layout.addLayout(top_layout); layout.addWidget(self.label_status, 1); widget.setLayout(layout)
        return widget

    def create_bl_panel(self):
        widget = QWidget(); layout = QVBoxLayout(); layout.setContentsMargins(5, 5, 5, 5); layout.addWidget(QLabel("실시간 데이터 로그"))
        hbox = QHBoxLayout(); hbox.setSpacing(10)
        all_log_widget = QWidget(); all_log_layout = QVBoxLayout(); all_log_layout.addWidget(QLabel("All Log")); self.all_log_table = QTableWidget(); self.all_log_table.setColumnCount(5); self.all_log_table.setHorizontalHeaderLabels(["ID", "Z1 Δ", "P Δ", "MD Δ", "Status"]); self.all_log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        all_log_layout.addWidget(self.all_log_table); all_log_widget.setLayout(all_log_layout)
        bad_log_widget = QWidget(); bad_log_layout = QVBoxLayout(); bad_log_layout.addWidget(QLabel("🚨 BAD Log (Alarm List) 🚨")); self.bad_log_table = QTableWidget(); self.bad_log_table.setColumnCount(6); self.bad_log_table.setHorizontalHeaderLabels(["Alarm ID", "Part ID", "Z1 Δ", "P Δ", "MD Δ", "Action"])
        header = self.bad_log_table.horizontalHeader(); header.setSectionResizeMode(QHeaderView.Stretch); header.setSectionResizeMode(5, QHeaderView.Fixed); self.bad_log_table.setColumnWidth(5, 80)
        bad_log_layout.addWidget(self.bad_log_table); bad_log_widget.setLayout(bad_log_layout)
        hbox.addWidget(all_log_widget); hbox.addWidget(bad_log_widget); layout.addLayout(hbox); widget.setLayout(layout)
        return widget

    def create_br_panel(self):
        widget = QWidget(); layout = QVBoxLayout(); layout.setContentsMargins(5, 5, 5, 5)
        self.bar_plot_widget = pg.PlotWidget(); self.bar_plot_widget.setTitle("양품/불량 집계"); self.bar_plot_widget.getAxis('bottom').setTicks([[(0.5, 'GOOD'), (1.5, 'BAD')]]); self.bar_plot_widget.setLabel('left', 'Count')
        self.bar_graph_good = pg.BarGraphItem(x=[0.5], height=[0], width=0.6, brush='#009E73'); self.bar_graph_bad = pg.BarGraphItem(x=[1.5], height=[0], width=0.6, brush='#D55E00')
        self.bar_plot_widget.addItem(self.bar_graph_good); self.bar_plot_widget.addItem(self.bar_graph_bad)
        self.bar_plot_widget.setYRange(0, 50); self.bar_plot_widget.setXRange(0, 2)
        self.good_text = pg.TextItem(text="", color=(0, 0, 0), anchor=(0.5, 0)); self.bad_text = pg.TextItem(text="", color=(0, 0, 0), anchor=(0.5, 0))
        self.bar_plot_widget.addItem(self.good_text); self.bar_plot_widget.addItem(self.bad_text)
        layout.addWidget(self.bar_plot_widget, 1)
        info_layout = QGridLayout(); info_layout.addWidget(QLabel("Total Count:"), 0, 0); info_layout.addWidget(QLabel("Yield Rate:"), 1, 0)
        self.label_br_total = QLabel("0"); info_layout.addWidget(self.label_br_total, 0, 1); self.label_br_yield = QLabel("0.00 %"); info_layout.addWidget(self.label_br_yield, 1, 1)
        layout.addLayout(info_layout); widget.setLayout(layout)
        return widget

    @pyqtSlot(dict)
    def update_ui(self, data):
        db_row = data['db_row']
        status = data['status']
        deviations = data['deviations']
        row_id = db_row['id']
        alarm_id = data.get('alarm_id')

        if data['prediction'] == -1: return

        self.total_count += 1
        if status == "GOOD": self.good_count += 1
        else: self.bad_count += 1
        yield_rate = (self.good_count / self.total_count) * 100 if self.total_count > 0 else 0

        # 그래프
        self.time_points.append(row_id)
        try:
            z1 = float(db_row.get('EX1.Z1_PV',0) or 0)
            p = float(db_row.get('EX1.MELT_P_PV',0) or 0)
            md = float(db_row.get('EX1.MD_PV',0) or 0)
        except (ValueError, TypeError):
            z1, p, md = 0.0, 0.0, 0.0
        self.param_z1_data.append(z1)
        self.param_melt_p_data.append(p)
        self.param_md_data.append(md)
        if len(self.time_points) > self.max_graph_points:
            self.time_points.pop(0); self.param_z1_data.pop(0); self.param_melt_p_data.pop(0); self.param_md_data.pop(0)
        self.line_z1.setData(self.time_points, self.param_z1_data)
        self.line_melt_p.setData(self.time_points, self.param_melt_p_data)
        self.line_md.setData(self.time_points, self.param_md_data)

        # 상태 패널 (Yield Rate만)
        self.label_yield_rate.setText(f"{yield_rate:.2f} %")
        if status == "GOOD":
            self.label_status.setText("GOOD")
            self.label_status.setStyleSheet("background-color:#C8E6C9; color:#2E7D32; border-radius:10px; padding: 10px;")
        else:
            self.label_status.setText(status)
            self.label_status.setStyleSheet("background-color:#FFCDD2; color:#C62828; border-radius:10px; padding: 10px;")

        # All Log 테이블
        self.all_log_table.insertRow(0)
        idx = QTableWidgetItem(str(row_id))
        z1_all = QTableWidgetItem(f"{deviations['Z1']:.2f}")
        p_all = QTableWidgetItem(f"{deviations['MELT_P']:.2f}")
        md_all = QTableWidgetItem(f"{deviations['MD']:.2f}")
        st = QTableWidgetItem(status)
        self.all_log_table.setItem(0,0,idx)
        self.all_log_table.setItem(0,1,z1_all)
        self.all_log_table.setItem(0,2,p_all)
        self.all_log_table.setItem(0,3,md_all)
        self.all_log_table.setItem(0,4,st)
        if self.all_log_table.rowCount() > 50: self.all_log_table.removeRow(50)

        # BAD Log 테이블 (불량 시)
        if status == "BAD" and alarm_id is not None:
            bg=QColor(255,205,210)
            idx.setBackground(bg); z1_all.setBackground(bg); p_all.setBackground(bg); md_all.setBackground(bg); st.setBackground(bg)
            self.bad_log_table.insertRow(0)
            aid=QTableWidgetItem(str(alarm_id))
            pid=QTableWidgetItem(str(row_id))
            z1b=QTableWidgetItem(f"{deviations['Z1']:.2f}")
            pb=QTableWidgetItem(f"{deviations['MELT_P']:.2f}")
            mdb=QTableWidgetItem(f"{deviations['MD']:.2f}")
            btn=QPushButton("확인")
            btn.setStyleSheet("background-color:#FFCDD2;")
            btn.clicked.connect(lambda _, b=btn, id=alarm_id: self.acknowledge_alarm(id, b))
            self.bad_log_table.setItem(0,0,aid); self.bad_log_table.setItem(0,1,pid); self.bad_log_table.setItem(0,2,z1b); self.bad_log_table.setItem(0,3,pb); self.bad_log_table.setItem(0,4,mdb)
            self.bad_log_table.setCellWidget(0,5,btn)
            aid.setBackground(bg); pid.setBackground(bg); z1b.setBackground(bg); pb.setBackground(bg); mdb.setBackground(bg)
            if self.bad_log_table.rowCount() > 50: self.bad_log_table.removeRow(50)

        # 막대 그래프
        counts = [self.good_count, self.bad_count]
        self.bar_graph_good.setOpts(height=[counts[0]])
        self.bar_graph_bad.setOpts(height=[counts[1]])
        self.good_text.setText(str(counts[0]))
        self.good_text.setPos(0.5, counts[0])
        self.bad_text.setText(str(counts[1]))
        self.bad_text.setPos(1.5, counts[1])
        max_cnt = max(counts) if max(counts) > 50 else 50
        self.bar_plot_widget.setYRange(0, max_cnt * 1.2)
        self.label_br_total.setText(f"{self.total_count}")
        self.label_br_yield.setText(f"{yield_rate:.2f} %")

    @pyqtSlot(int, QPushButton)
    def acknowledge_alarm(self, alarm_id, button_ref):
        # 현재 행 인덱스 찾기
        row_idx = self.bad_log_table.indexAt(button_ref.pos()).row()
        if row_idx < 0:
            print(f"Error: Invalid row index for alarm {alarm_id}.")
            return

        # 알람 ID 재확인 (정확성 위해)
        # [V17.1] 올바른 try...except 구조
        try:
            item = self.bad_log_table.item(row_idx, 0) # Alarm ID 컬럼(0)
            # 아이템이 존재하고, 텍스트를 정수로 변환했을 때 alarm_id와 일치하는지 확인
            if item is None or int(item.text()) != alarm_id:
                print(f"Error: Row {row_idx} ID mismatch for alarm {alarm_id}.")
                return
        except (ValueError, TypeError): # int 변환 실패 등
            print(f"Error: Cannot verify alarm ID at row {row_idx}.")
            return

        print(f"Toggling Alarm: {alarm_id} @ row {row_idx}")
        db = None # finally에서 사용 위해 선언

        # [V17.1] 올바른 try...except...finally 구조
        try:
            db = SessionLocal() # DB 세션 시작

            # 현재 알람 상태 조회
            current_status = db.execute(text("SELECT status FROM alarm_log WHERE id = :aid"), {"aid": alarm_id}).scalar_one_or_none()
            if current_status is None:
                print(f"Error: Alarm {alarm_id} not found in DB.")
                button_ref.setText("오류")
                button_ref.setEnabled(False)
                return # 함수 종료

            # 상태 토글 및 DB 업데이트
            if current_status == 'New':
                new_status = 'Acknowledged'
                update_res = db.execute(text("UPDATE alarm_log SET status=:s, ack_by=:u, ack_timestamp=NOW() WHERE id=:aid RETURNING ack_timestamp"),
                                        {"s":new_status, "u":"Operator", "aid":alarm_id})
                db.commit()
                ack_time = update_res.scalar_one_or_none()

                # UI 업데이트 (성공 시)
                if ack_time:
                    print(f"Alarm {alarm_id} acknowledged.")
                    button_ref.setText("취소")
                    button_ref.setStyleSheet("background-color: #E0E0E0;") # 회색
                    bg_color=QColor(Qt.white) # 배경 흰색
                else: # 업데이트 실패
                    print(f"Failed to acknowledge alarm {alarm_id}.")
                    return # 함수 종료

            elif current_status == 'Acknowledged':
                new_status = 'New'
                db.execute(text("UPDATE alarm_log SET status=:s, ack_by=NULL, ack_timestamp=NULL WHERE id=:aid"),
                           {"s":new_status, "aid":alarm_id})
                db.commit()
                ack_time = "Unack" # UI 업데이트 구분용

                print(f"Alarm {alarm_id} unacknowledged.")
                button_ref.setText("확인")
                button_ref.setStyleSheet("background-color: #FFCDD2;") # 빨간색
                bg_color=QColor(255, 205, 210) # 배경 빨간색
            else: # 알 수 없는 상태
                print(f"Unknown DB status '{current_status}' for alarm {alarm_id}.")
                return # 함수 종료

            # 테이블 행 배경색 업데이트
            for c in range(self.bad_log_table.columnCount() -1): # 마지막 Action 컬럼 제외
                itm = self.bad_log_table.item(row_idx, c)
                if itm:
                    itm.setBackground(bg_color)

        # DB 오류 처리
        except exc.SQLAlchemyError as e:
            print(f"DB Ack Toggle Err: {e}")
            self.show_db_error(f"Alarm Toggle Fail: {e}")
            if db: db.rollback() # 오류 시 롤백
        # 기타 오류 처리 (버튼 위치 찾기 실패 등)
        except Exception as e:
            print(f"Ack Toggle Err: {e}")
        finally:
            # DB 세션 종료
            if db:
                db.close()

    @pyqtSlot()
    def poll_control_status(self):
        db = None
        status = 'Unknown'
        cur_cnt = 0
        tgt_cnt = self.current_target_count # 이전 값으로 초기화
        wo = self.current_wo_label.text() # 현재 UI 값으로 초기화

        # [V17.1] 올바른 try...except...finally 구조
        try:
            db = SessionLocal()
            ctrl = db.execute(text("SELECT status, current_count, target_count, work_order FROM process_control WHERE id = 1")).first()

            if ctrl:
                status = ctrl.status
                cur_cnt = ctrl.current_count
                tgt_cnt = ctrl.target_count
                wo = ctrl.work_order
                self.current_target_count = tgt_cnt # 최신 목표 값 업데이트

                # 현재 WO 라벨 업데이트 (DB 값이 None이 �