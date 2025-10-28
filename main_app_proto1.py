"""
플랫폼 1: 실시간 품질 예측 모니터링 플랫폼 (main_app.py)

- V10: (사용자 제안)
  1. 좌측 하단(BL) 테이블들의 컬럼이 가로로 꽉 차도록 수정
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
    QPushButton, QHeaderView # <<< [변경] QHeaderView 임포트
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QColor

import pyqtgraph as pg

# --- 1. 환경 변수 및 설정 로드 ---
print("환경 변수 로딩...")
load_dotenv() 

DATA_PATH = os.getenv('DATA_PATH')
MODEL_PATH = os.getenv('MODEL_PATH')

OPTIMAL_VALUES = {
    'EX1.Z1_PV': 210.12,
    'EX1.MELT_P_PV': 6.86,
    'EX1.MD_PV': 71.32
}
print(f"데이터 경로: {DATA_PATH}")
print(f"모델 경로: {MODEL_PATH}")


# --- 2. ML Core (백엔드 QThread) ---
class MlCoreWorker(QThread):
    newData = pyqtSignal(dict)
    
    def __init__(self, model_path, data_path):
        super().__init__()
        self.model_path = model_path
        self.data_path = data_path
        self.running = True
        self.model_loaded = False
        self.data_loaded = False

        try:
            pipeline_data = joblib.load(self.model_path)
            self.scaler = pipeline_data['scaler']
            self.model = pipeline_data['model']
            self.features_order = pipeline_data['features']
            self.model_loaded = True
            print("ML Core Worker: 모델 로드 성공.")
        except FileNotFoundError:
            print(f"오류: {self.model_path}를 찾을 수 없습니다.")
        
        try:
            self.df = pd.read_csv(self.data_path)
            self.df.dropna(subset=['passorfail'], inplace=True)
            self.data_loaded = True
            print(f"ML Core Worker: {self.data_path} 로드 성공.")
        except FileNotFoundError:
            print(f"오류: {self.data_path}를 찾을 수 없습니다.")

    def run(self):
        if not self.running or not self.model_loaded or not self.data_loaded:
            print("ML Core Worker: 모델/데이터가 로드되지 않아 중지됨.")
            return
            
        good_count = 0
        bad_count = 0
        
        for index, row in self.df.iterrows():
            if not self.running:
                break
                
            total_count = index + 1
            
            if row['EX1.MD_TQ'] == 0:
                prediction = 1
            else:
                current_features_df = pd.DataFrame([row[self.features_order]])
                current_features_scaled = self.scaler.transform(current_features_df)
                prediction = self.model.predict(current_features_scaled)[0]
            
            dev_z1 = row['EX1.Z1_PV'] - OPTIMAL_VALUES['EX1.Z1_PV']
            dev_melt_p = row['EX1.MELT_P_PV'] - OPTIMAL_VALUES['EX1.MELT_P_PV']
            dev_md = row['EX1.MD_PV'] - OPTIMAL_VALUES['EX1.MD_PV']

            if prediction == 0:
                good_count += 1; status = "GOOD"
            else:
                bad_count += 1; status = "BAD"
            yield_rate = (good_count / total_count) * 100

            data_packet = {
                'index': total_count, 'status': status,
                'yield_rate': yield_rate, 'good_count': good_count, 'bad_count': bad_count,
                'total_count': total_count,
                'params': {
                    'EX1.Z1_PV': row['EX1.Z1_PV'],
                    'EX1.MELT_P_PV': row['EX1.MELT_P_PV'],
                    'EX1.MD_PV': row['EX1.MD_PV']
                },
                'deviations': {'Z1': dev_z1, 'MELT_P': dev_melt_p, 'MD': dev_md}
            }
            
            self.newData.emit(data_packet)
            time.sleep(0.05)
            
        print("ML Core Worker: 시뮬레이션 완료.")

    def stop(self):
        self.running = False
        self.wait()

# --- 3. 플랫폼 1: 메인 UI (PyQt) ---
class RealtimeDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("실시간 품질 예측 모니터링 플랫폼 (Platform 1)")
        self.setGeometry(100, 100, 1600, 900)
        
        self.time_points = []
        self.param_z1_data = []
        self.param_melt_p_data = []
        self.param_md_data = []
        self.max_graph_points = 200
        
        self.initUI()
        
        self.worker = MlCoreWorker(MODEL_PATH, DATA_PATH)
        self.worker.newData.connect(self.update_ui)
        self.worker.start()

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
        # (변경 사항 없음)
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
        self.apply_button.clicked.connect(self.restart_simulation) 
        model_selection_layout.addWidget(self.apply_button)
        header_layout.addLayout(model_selection_layout)
        return header_layout

    def create_tl_panel(self):
        # (변경 사항 없음)
        pg.setConfigOptions(antialias=True, background='w', foreground='k')
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Sensor Value')
        self.plot_widget.setLabel('bottom', 'Time (Data Points)')
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
        # (변경 사항 없음)
        widget = QWidget()
        layout = QVBoxLayout()
        
        top_layout = QGridLayout()
        top_layout.addWidget(QLabel("Part No.:"), 0, 0)
        top_layout.addWidget(QLabel("Yield Rate:"), 1, 0)
        
        self.label_part_no = QLabel("0")
        self.label_part_no.setFont(QFont('Arial', 18, QFont.Bold))
        self.label_yield_rate = QLabel("0.00 %")
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

    # --- [V10 변경] ---
    def create_bl_panel(self):
        """(좌측 하단) 2분할 로그 패널 (컬럼 꽉 채우기)"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("실시간 데이터 로그"))
        
        hbox = QHBoxLayout()
        
        # 1. 왼쪽: 전체 로그 테이블
        all_log_widget = QWidget()
        all_log_layout = QVBoxLayout()
        all_log_layout.addWidget(QLabel("All Log"))
        
        self.all_log_table = QTableWidget()
        self.all_log_table.setColumnCount(5)
        self.all_log_table.setHorizontalHeaderLabels(["No.", "Z1 편차", "P 편차", "MD 편차", "Status"])
        # [수정] ResizeToContents -> Stretch
        self.all_log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        all_log_layout.addWidget(self.all_log_table)
        all_log_widget.setLayout(all_log_layout)
        
        # 2. 오른쪽: 불량 로그 테이블
        bad_log_widget = QWidget()
        bad_log_layout = QVBoxLayout()
        bad_log_layout.addWidget(QLabel("🚨 BAD Log 🚨"))
        
        self.bad_log_table = QTableWidget()
        self.bad_log_table.setColumnCount(4)
        self.bad_log_table.setHorizontalHeaderLabels(["No.", "Z1 편차", "P 편차", "MD 편차"])
        # [수정] ResizeToContents -> Stretch
        self.bad_log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        bad_log_layout.addWidget(self.bad_log_table)
        bad_log_widget.setLayout(bad_log_layout)
        
        hbox.addWidget(all_log_widget)
        hbox.addWidget(bad_log_widget)
        
        layout.addLayout(hbox)
        widget.setLayout(layout)
        return widget
    # --- [변경 끝] ---

    def create_br_panel(self):
        # (변경 사항 없음 - V9의 버그 수정 코드)
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
        
    def update_ui(self, data):
        """[핵심] ML Core(스레드)로부터 신호를 받으면 4개 패널 모두 업데이트"""
        
        # --- (TL) 꺾은선 그래프 (변경 없음) ---
        self.time_points.append(data['index'])
        self.param_z1_data.append(data['params']['EX1.Z1_PV'])
        self.param_melt_p_data.append(data['params']['EX1.MELT_P_PV'])
        self.param_md_data.append(data['params']['EX1.MD_PV'])
        
        if len(self.time_points) > self.max_graph_points:
            self.time_points.pop(0); self.param_z1_data.pop(0); self.param_melt_p_data.pop(0); self.param_md_data.pop(0)
            
        self.line_z1.setData(self.time_points, self.param_z1_data)
        self.line_melt_p.setData(self.time_points, self.param_melt_p_data)
        self.line_md.setData(self.time_points, self.param_md_data)
        
        # --- (TR) 실시간 상태 (변경 없음) ---
        self.label_part_no.setText(f"{data['total_count']}")
        self.label_yield_rate.setText(f"{data['yield_rate']:.2f} %")
        if data['status'] == "GOOD":
            self.label_status.setText("GOOD"); self.label_status.setStyleSheet("background-color: #C8E6C9; color: #2E7D32; border-radius: 10px;")
        else:
            self.label_status.setText("BAD"); self.label_status.setStyleSheet("background-color: #FFCDD2; color: #C62828; border-radius: 10px;")

        # --- (BL) 2분할 테이블 (변경 없음) ---
        self.all_log_table.insertRow(0)
        item_index_all = QTableWidgetItem(str(data['index']))
        item_z1_all = QTableWidgetItem(f"{data['deviations']['Z1']:.2f}")
        item_p_all = QTableWidgetItem(f"{data['deviations']['MELT_P']:.2f}")
        item_md_all = QTableWidgetItem(f"{data['deviations']['MD']:.2f}")
        item_status_all = QTableWidgetItem(data['status'])
        
        self.all_log_table.setItem(0, 0, item_index_all)
        self.all_log_table.setItem(0, 1, item_z1_all)
        self.all_log_table.setItem(0, 2, item_p_all)
        self.all_log_table.setItem(0, 3, item_md_all)
        self.all_log_table.setItem(0, 4, item_status_all)

        if self.all_log_table.rowCount() > 50: self.all_log_table.removeRow(50)
            
        if data['status'] == "BAD":
            bg_color = QColor(255, 205, 210)
            item_index_all.setBackground(bg_color)
            item_z1_all.setBackground(bg_color)
            item_p_all.setBackground(bg_color)
            item_md_all.setBackground(bg_color)
            item_status_all.setBackground(bg_color)
            
            self.bad_log_table.insertRow(0)
            item_index_bad = QTableWidgetItem(str(data['index']))
            item_z1_bad = QTableWidgetItem(f"{data['deviations']['Z1']:.2f}")
            item_p_bad = QTableWidgetItem(f"{data['deviations']['MELT_P']:.2f}")
            item_md_bad = QTableWidgetItem(f"{data['deviations']['MD']:.2f}")
            
            self.bad_log_table.setItem(0, 0, item_index_bad)
            self.bad_log_table.setItem(0, 1, item_z1_bad)
            self.bad_log_table.setItem(0, 2, item_p_bad)
            self.bad_log_table.setItem(0, 3, item_md_bad)
            
            item_index_bad.setBackground(bg_color)
            item_z1_bad.setBackground(bg_color)
            item_p_bad.setBackground(bg_color)
            item_md_bad.setBackground(bg_color)

            if self.bad_log_table.rowCount() > 50: self.bad_log_table.removeRow(50)
            
        # --- (BR) 막대 그래프 + 텍스트 업데이트 (V9 버그 수정 코드) ---
        counts = [data['good_count'], data['bad_count']]
        
        self.bar_graph_good.setOpts(height=[counts[0]]) # V9 수정: height
        self.bar_graph_bad.setOpts(height=[counts[1]])  # V9 수정: height
        
        self.good_text.setText(str(counts[0]))
        self.good_text.setPos(0.5, counts[0])
        
        self.bad_text.setText(str(counts[1]))
        self.bad_text.setPos(1.5, counts[1])
        
        max_count = max(counts) if max(counts) > 50 else 50
        self.bar_plot_widget.setYRange(0, max_count * 1.2)
        
        self.label_br_total.setText(f"{data['total_count']}")
        self.label_br_yield.setText(f"{data['yield_rate']:.2f} %")
        
    def restart_simulation(self):
        # (변경 사항 없음)
        print("시뮬레이션 재시작 요청...")
        if self.worker.isRunning():
            self.worker.stop()
        
        self.time_points.clear(); self.param_z1_data.clear(); self.param_melt_p_data.clear(); self.param_md_data.clear()
        self.all_log_table.setRowCount(0)
        self.bad_log_table.setRowCount(0)
        
        selected_model = self.model_combo.currentText()
        print(f"'{selected_model}' 모델로 재시작합니다.")
        
        self.worker = MlCoreWorker(MODEL_PATH, DATA_PATH)
        self.worker.newData.connect(self.update_ui)
        self.worker.start()

    def closeEvent(self, event):
        print("윈도우 종료. ML Core 스레드를 중지합니다.")
        self.worker.stop()
        event.accept()

# --- 4. 애플리케이션 실행 ---
if __name__ == "__main__":
    if not all([DATA_PATH, MODEL_PATH]):
        print("오류: .env 파일에 DATA_PATH 또는 MODEL_PATH가 설정되지 않았습니다.")
        print("프로그램을 종료합니다.")
    else:
        app = QApplication(sys.argv)
        window = RealtimeDashboard()
        window.show()
        sys.exit(app.exec_())