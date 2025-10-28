"""
플랫폼 0: 공정 시뮬레이터 (generator_app.py)

- V2:
  1. Pause/Resume 로직 개선 (즉각 반응하도록 수정)
  2. (DB는 TIMESTAMPTZ(3)으로 변경됨)
"""
import sys
import os
import pandas as pd
import numpy as np
import random
import time
import datetime
from dotenv import load_dotenv

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QMutex, QWaitCondition
from PyQt5.QtGui import QFont

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# --- 1. 환경 변수 및 DB 설정 로드 ---
print("환경 변수 로딩...")
load_dotenv() 

DATABASE_URL = os.getenv('DATABASE_URL')
DATA_PATH = os.getenv('DATA_PATH') 

if not DATABASE_URL or not DATA_PATH:
    print("오류: .env 파일에 DATABASE_URL 또는 DATA_PATH가 설정되지 않았습니다.")
    sys.exit()

print(f"DB URL: {DATABASE_URL}")
print(f"Base Data Path: {DATA_PATH}")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- 2. Generator Worker (백엔드 QThread) ---
class GeneratorWorker(QThread):
    update_status = pyqtSignal(str) 
    process_finished = pyqtSignal() 
    db_error = pyqtSignal(str)      
    
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.running = True
        
        # [V2 변경] Pause 상태 관리를 위한 Mutex와 WaitCondition
        self.pause_mutex = QMutex()
        self.pause_cond = QWaitCondition()
        self._is_paused = False # 내부 pause 상태 플래그
        
        try:
            self.base_df = pd.read_csv(DATA_PATH)
            self.numeric_cols = self.base_df.select_dtypes(include=np.number).columns
            self.numeric_cols = self.numeric_cols.drop('passorfail', errors='ignore') 
            self.base_stats = self.base_df[self.numeric_cols].agg(['mean', 'std'])
            print("Generator Worker: Base data profile loaded.")
        except FileNotFoundError:
            print(f"오류: Base data file not found at {DATA_PATH}")
            self.base_df = None 
        except Exception as e:
            print(f"Base data 로딩 중 오류: {e}")
            self.base_df = None

    # [V2 변경] is_paused 속성 (Mutex 보호)
    @property
    def is_paused(self):
        self.pause_mutex.lock()
        paused = self._is_paused
        self.pause_mutex.unlock()
        return paused

    @is_paused.setter
    def is_paused(self, value):
        self.pause_mutex.lock()
        self._is_paused = value
        self.pause_mutex.unlock()
        if not value: # False (Resume)이면 대기 중인 스레드를 깨움
            self.pause_cond.wakeAll()

    def run(self):
        current_count = 0
        target_count = self.settings['target_count']
        cycle_time = self.settings['cycle_time']
        defect_rate = self.settings['defect_rate'] / 100.0
        
        try:
            db = SessionLocal() 

            db.execute(text("""
                UPDATE process_control SET 
                status = :status, work_order = :wo, product_name = :prod, 
                target_count = :target, current_count = :current, 
                cycle_time_seconds = :cycle, defect_rate_percent = :defect, 
                last_updated = NOW() 
                WHERE id = 1
            """), {
                "status": "Running", "wo": self.settings['work_order'], 
                "prod": self.settings['product_name'], "target": target_count, 
                "current": 0, "cycle": cycle_time, "defect": self.settings['defect_rate']
            })
            db.commit()
            
            self.update_status.emit(f"Running: 0 / {target_count}")

            while self.running and current_count < target_count:
                # --- [V2 변경] 일시정지 로직 ---
                self.pause_mutex.lock()
                while self._is_paused:
                    # DB 상태도 확인 (외부에서 Paused 시켰을 경우 대비)
                    db_status = db.execute(text("SELECT status FROM process_control WHERE id=1")).scalar()
                    if db_status != 'Paused':
                        print("DB status is not Paused, resuming automatically...")
                        self._is_paused = False # 내부 상태도 동기화
                        db.execute(text("UPDATE process_control SET status = 'Running', last_updated = NOW() WHERE id = 1"))
                        db.commit()
                        self.update_status.emit(f"Running: {current_count} / {target_count}")
                        break # Pause 루프 탈출
                    
                    # Mutex를 잠시 풀고 대기 상태로 진입 (resume()이 wakeAll() 호출 시 깨어남)
                    self.pause_cond.wait(self.pause_mutex, 1000) # 1초 타임아웃
                    if not self.running: break # 강제 종료 시 탈출
                self.pause_mutex.unlock()
                if not self.running: break
                # --- 일시정지 로직 끝 ---

                wait_time = cycle_time * random.uniform(0.9, 1.1)
                time.sleep(wait_time)
                
                new_data = {}
                if self.base_df is not None:
                    for col in self.numeric_cols:
                        mean = self.base_stats.loc['mean', col]
                        std = self.base_stats.loc['std', col]
                        noise = random.uniform(-0.5, 0.5) * std if std > 0 else 0 
                        new_data[col] = round(mean + noise, 5)
                else: 
                    for col in self.numeric_cols: new_data[col] = 0.0

                is_defect = random.random() < defect_rate
                passorfail = 1 if is_defect else 0
                
                if is_defect:
                    new_data['EX1.MD_TQ'] = 0.0
                else:
                     mean_tq = self.base_stats.loc['mean', 'EX1.MD_TQ'] if self.base_df is not None else 72
                     new_data['EX1.MD_TQ'] = round(mean_tq + random.uniform(-1, 1), 5) 

                quoted_cols = [f'"{col}"' for col in self.numeric_cols]
                placeholders = [f':{col.replace(".", "_")}' for col in self.numeric_cols] 
                param_dict = {col.replace(".", "_"): float(new_data[col]) for col in self.numeric_cols} 
                param_dict["wo"] = self.settings['work_order']
                param_dict["pf"] = passorfail
                
                insert_sql = text(f"""
                    INSERT INTO process_data (work_order, {", ".join(quoted_cols)}, passorfail) 
                    VALUES (:wo, {", ".join(placeholders)}, :pf)
                """)
                db.execute(insert_sql, param_dict)
                db.commit() 

                current_count += 1

                db.execute(text("""
                    UPDATE process_control SET current_count = :current, last_updated = NOW() WHERE id = 1
                """), {"current": current_count})
                db.commit()

                self.update_status.emit(f"Running: {current_count} / {target_count}")

            if self.running: 
                final_status = "Finished"
                self.update_status.emit(f"Finished: {current_count} / {target_count}")
            else: 
                final_status = "Stopped"
                self.update_status.emit(f"Stopped at {current_count} / {target_count}")

            db.execute(text("UPDATE process_control SET status = :status, last_updated = NOW() WHERE id = 1"), 
                       {"status": final_status})
            db.commit()
            
            self.process_finished.emit() 

        except Exception as e:
            self.db_error.emit(f"DB 오류 발생: {e}")
            print(f"Generator Worker DB Error: {e}")
            try:
                db.execute(text("UPDATE process_control SET status = 'Error', last_updated = NOW() WHERE id = 1"))
                db.commit()
            except: pass 
        finally:
            if 'db' in locals() and db:
                db.close()
            print("Generator Worker: Thread finished.")

    def stop(self):
        self.running = False
        self.is_paused = False # Pause 상태 강제 해제 (WaitCondition에서 깨어나도록)
        print("Generator Worker: Stopping...")

    # [V2 변경] Pause 메소드
    def pause(self):
        if not self.is_paused:
            self.is_paused = True # 내부 플래그 설정
            # DB 상태도 Paused로 변경
            try:
                db = SessionLocal()
                db.execute(text("UPDATE process_control SET status = 'Paused', last_updated = NOW() WHERE id = 1"))
                db.commit()
                db.close()
                self.update_status.emit("Paused")
                print("Generator Worker: Paused.")
            except Exception as e:
                 self.db_error.emit(f"DB Pause 오류: {e}")

    # [V2 변경] Resume 메소드
    def resume(self):
        if self.is_paused:
            self.is_paused = False # 내부 플래그 해제 (즉시 run 루프가 재개됨)
            # DB 상태도 Running으로 변경
            try:
                db = SessionLocal()
                db.execute(text("UPDATE process_control SET status = 'Running', last_updated = NOW() WHERE id = 1"))
                db.commit()
                # 현재 카운트 가져와서 상태 업데이트
                counts = db.execute(text("SELECT current_count, target_count FROM process_control WHERE id = 1")).first()
                db.close()
                if counts:
                    self.update_status.emit(f"Running: {counts.current_count} / {counts.target_count}")
                print("Generator Worker: Resumed.")
            except Exception as e:
                 self.db_error.emit(f"DB Resume 오류: {e}")


# --- 3. 메인 UI (PyQt) ---
class GeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("플랫폼 0: 공정 시뮬레이터")
        self.setGeometry(100, 100, 400, 400)
        
        self.worker = None 

        self.initUI()
        self.check_initial_db_status() 

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel("공정 이름 (Work Order):"), 0, 0)
        self.wo_input = QLineEdit("WO-" + datetime.datetime.now().strftime("%Y%m%d%H%M"))
        grid_layout.addWidget(self.wo_input, 0, 1)
        grid_layout.addWidget(QLabel("제품명:"), 1, 0)
        self.product_input = QLineEdit("A-Pipe")
        grid_layout.addWidget(self.product_input, 1, 1)
        grid_layout.addWidget(QLabel("공정 책임자:"), 2, 0)
        self.operator_input = QLineEdit("Operator")
        grid_layout.addWidget(self.operator_input, 2, 1) 
        grid_layout.addWidget(QLabel("목표 생산량 (개):"), 3, 0)
        self.target_count_input = QSpinBox()
        self.target_count_input.setRange(1, 100000); self.target_count_input.setValue(2000)
        grid_layout.addWidget(self.target_count_input, 3, 1)
        grid_layout.addWidget(QLabel("사이클 타임 (초):"), 4, 0)
        self.cycle_time_input = QDoubleSpinBox()
        self.cycle_time_input.setRange(0.1, 60.0); self.cycle_time_input.setValue(5.0); self.cycle_time_input.setSingleStep(0.1)
        grid_layout.addWidget(self.cycle_time_input, 4, 1)
        grid_layout.addWidget(QLabel("불량률 (%):"), 5, 0)
        self.defect_rate_input = QDoubleSpinBox()
        self.defect_rate_input.setRange(0.0, 100.0); self.defect_rate_input.setValue(0.5); self.defect_rate_input.setSingleStep(0.1)
        grid_layout.addWidget(self.defect_rate_input, 5, 1)
        main_layout.addLayout(grid_layout)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("공정 시작")
        self.pause_button = QPushButton("일시 정지") # [V2] 이름 통일
        self.stop_button = QPushButton("강제 종료")
        
        self.start_button.clicked.connect(self.start_process)
        self.pause_button.clicked.connect(self.pause_resume_process) # [V2] 연결 함수 변경
        self.stop_button.clicked.connect(self.stop_process)
        
        button_layout.addWidget(self.start_button); button_layout.addWidget(self.pause_button); button_layout.addWidget(self.stop_button)
        main_layout.addLayout(button_layout)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setFont(QFont('Arial', 14, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def check_initial_db_status(self):
        """앱 시작 시 DB 상태 확인 및 UI 동기화"""
        try:
            db = SessionLocal()
            control = db.execute(text("SELECT status, current_count, target_count FROM process_control WHERE id = 1")).first()
            db.close()
            
            if control and control.status in ["Running", "Paused"]:
                 QMessageBox.warning(self, "진행 중인 공정 감지", 
                                     f"DB에 '{control.status}' 상태의 공정이 남아있습니다.\n"
                                     f"({control.current_count}/{control.target_count} 진행됨)\n"
                                     "UI 상태를 동기화합니다.")
                 self.status_label.setText(f"Status: {control.status} (DB Sync)")
                 self.set_controls_state(running=True, paused=(control.status == 'Paused')) # [V2] paused 상태 추가

        except Exception as e:
            self.show_db_error(f"초기 DB 상태 확인 중 오류: {e}")

    def start_process(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "오류", "이미 공정이 진행 중입니다.")
            return

        settings = {
            'work_order': self.wo_input.text(), 'product_name': self.product_input.text(),
            'target_count': self.target_count_input.value(), 'cycle_time': self.cycle_time_input.value(),
            'defect_rate': self.defect_rate_input.value()
        }
        if not settings['work_order'] or not settings['product_name']:
             QMessageBox.warning(self, "입력 오류", "공정 이름과 제품명을 입력하세요."); return

        print("Starting process with settings:", settings)
        
        self.worker = GeneratorWorker(settings)
        self.worker.update_status.connect(self.update_status_label)
        self.worker.process_finished.connect(self.on_process_finished)
        self.worker.db_error.connect(self.show_db_error)
        self.worker.start()
        self.set_controls_state(running=True, paused=False)

    # [V2 변경] Pause/Resume 통합 함수
    def pause_resume_process(self):
        if self.worker and self.worker.isRunning():
            if not self.worker.is_paused:
                self.worker.pause()
                self.pause_button.setText("다시 시작")
            else:
                self.worker.resume()
                self.pause_button.setText("일시 정지")

    def stop_process(self):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, '공정 종료 확인', '정말로 공정을 강제 종료하시겠습니까?', 
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.set_controls_state(running=False, paused=False) 
        else:
             try: # DB 리셋 로직 유지
                 db = SessionLocal()
                 db.execute(text("UPDATE process_control SET status = 'Idle', current_count = 0, last_updated = NOW() WHERE id = 1"))
                 db.commit(); db.close()
                 self.status_label.setText("Status: Idle (Reset)")
                 self.set_controls_state(running=False, paused=False)
             except Exception as e:
                 self.show_db_error(f"DB 리셋 중 오류: {e}")

    def update_status_label(self, status_text):
        self.status_label.setText(f"Status: {status_text}")

    def on_process_finished(self):
        self.set_controls_state(running=False, paused=False)
        QMessageBox.information(self, "공정 완료", "설정된 공정이 완료되었습니다.")

    # [V2 변경] paused 상태 반영
    def set_controls_state(self, running, paused):
        self.start_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)
        
        # Pause 상태에 따라 버튼 텍스트 변경
        if running and paused:
            self.pause_button.setText("다시 시작")
        else:
             self.pause_button.setText("일시 정지")

        for i in range(self.findChild(QGridLayout).count()): 
             widget = self.findChild(QGridLayout).itemAt(i).widget()
             if isinstance(widget, (QLineEdit, QSpinBox, QDoubleSpinBox)):
                 widget.setEnabled(not running)

    def show_db_error(self, error_message):
         QMessageBox.critical(self, "데이터베이스 오류", error_message)
         self.set_controls_state(running=False, paused=False) 
         self.status_label.setText("Status: DB Error!")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        event.accept()

# --- 4. 애플리케이션 실행 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GeneratorApp()
    window.show()
    sys.exit(app.exec_())