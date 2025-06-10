import sys
import json
from datetime import datetime
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QIcon
from face_recognition import AdvancedFaceRecognition
class AttendanceController:
    """Controller to interface with your attendance module"""
    def __init__(self):
        self.fr=AdvancedFaceRecognition()
        self.is_running = False
        self.start_time = None
        self.session_file = Path("attendance_session.json")
        self.load_session()
    
    def start_attendance(self):
        """Start the attendance tracking"""
        try:
            self.is_running = True
            self.start_time = datetime.now()
            self.save_session()
            
            # Here you would call your actual attendance module's start function
            # Example: your_attendance_module.start()
            self.fr.run()
            # return True, "Attendance tracking started successfully"
            
        except Exception as e:
            return False, f"Failed to start: {str(e)}"
    
    def stop_attendance(self):
        """Stop the attendance tracking"""
        try:
            self.is_running = False
            end_time = datetime.now()
            
            # Calculate session duration if there was a start time
            duration = None
            if self.start_time:
                duration = end_time - self.start_time
            
            self.clear_session()
            
            # Here you would call your actual attendance module's stop function
            # Example: your_attendance_module.stop()
            
            duration_str = f" (Duration: {duration})" if duration else ""
            self.fr.stop()
            return True, f"Attendance tracking stopped{duration_str}"
        except Exception as e:
            return False, f"Failed to stop: {str(e)}"
    
    def save_session(self):
        """Save current session state"""
        session_data = {
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None
        }
        with open(self.session_file, 'w') as f:
            json.dump(session_data, f)
    
    def load_session(self):
        """Load previous session state"""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                self.is_running = data.get("is_running", False)
                start_time_str = data.get("start_time")
                if start_time_str:
                    self.start_time = datetime.fromisoformat(start_time_str)
            except Exception:
                self.is_running = False
                self.start_time = None
    
    def clear_session(self):
        """Clear session file"""
        if self.session_file.exists():
            self.session_file.unlink()
        self.start_time = None

class ModernButton(QPushButton):
    def __init__(self, text, button_type="primary"):
        super().__init__(text)
        self.setMinimumHeight(50)
        self.setMinimumWidth(200)
        self.set_style(button_type)
    
    def set_style(self, button_type):
        if button_type == "start":
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #10B981, stop:1 #059669);
                    border: none;
                    border-radius: 12px;
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 15px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #059669, stop:1 #047857);
                }
                QPushButton:pressed {
                    background: #047857;
                }
                QPushButton:disabled {
                    background: #9CA3AF;
                }
            """)
        elif button_type == "stop":
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #EF4444, stop:1 #DC2626);
                    border: none;
                    border-radius: 12px;
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 15px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #DC2626, stop:1 #B91C1C);
                }
                QPushButton:pressed {
                    background: #B91C1C;
                }
                QPushButton:disabled {
                    background: #9CA3AF;
                }
            """)

class AttendanceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.controller = AttendanceController()
        self.setup_ui()
        self.setup_timer()
        self.update_ui_state()
    
    def setup_ui(self):
        self.setWindowTitle("Attendance Manager")
        self.setFixedSize(600, 400)
        
        # Apply main window styling
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(50, 50, 50, 50)
        
        # Title
        title_label = QLabel("Attendance Management")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 32px;
                font-weight: bold;
                margin-bottom: 20px;
            }
        """)
        main_layout.addWidget(title_label)
        
        # Status card
        self.status_frame = QFrame()
        self.status_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 16px;
                padding: 20px;
            }
        """)
        status_layout = QVBoxLayout(self.status_frame)
        
        # Status label
        self.status_label = QLabel("Status: Stopped")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #374151;
                margin-bottom: 10px;
            }
        """)
        status_layout.addWidget(self.status_label)
        
        # Time label
        self.time_label = QLabel("Ready to start")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #6B7280;
            }
        """)
        status_layout.addWidget(self.time_label)
        
        main_layout.addWidget(self.status_frame)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)
        
        # Start button
        self.start_button = ModernButton("🚀 START", "start")
        self.start_button.clicked.connect(self.start_attendance)
        button_layout.addWidget(self.start_button)
        
        # Stop button
        self.stop_button = ModernButton("⏹️ STOP", "stop")
        self.stop_button.clicked.connect(self.stop_attendance)
        button_layout.addWidget(self.stop_button)
        
        main_layout.addLayout(button_layout)
        
        # Add stretch to center content
        main_layout.addStretch()
    
    def setup_timer(self):
        """Setup timer to update running time display"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time_display)
        self.timer.start(1000)  # Update every second
    
    def update_ui_state(self):
        """Update UI based on current state"""
        if self.controller.is_running:
            self.status_label.setText("Status: Running")
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    font-weight: bold;
                    color: #059669;
                    margin-bottom: 10px;
                }
            """)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        else:
            self.status_label.setText("Status: Stopped")
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    font-weight: bold;
                    color: #374151;
                    margin-bottom: 10px;
                }
            """)
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.time_label.setText("Ready to start")
    
    def update_time_display(self):
        """Update the time display if running"""
        if self.controller.is_running and self.controller.start_time:
            duration = datetime.now() - self.controller.start_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_label.setText(f"Running: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    
    def start_attendance(self):
        """Handle start button click"""
        success, message = self.controller.start_attendance()
        
        if success:
            self.update_ui_state()
            self.show_message("Success", message, QMessageBox.Information)
        else:
            self.show_message("Error", message, QMessageBox.Critical)
    
    def stop_attendance(self):
        """Handle stop button click"""
        success, message = self.controller.stop_attendance()
        
        if success:
            self.update_ui_state()
            self.show_message("Success", message, QMessageBox.Information)
        else:
            self.show_message("Error", message, QMessageBox.Critical)
    
    def show_message(self, title, message, msg_type):
        """Show message box"""
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(msg_type)
        msg_box.exec()
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.controller.is_running:
            reply = QMessageBox.question(
                self, 'Close Application',
                'Attendance tracking is still running. Stop it before closing?',
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Yes:
                self.controller.stop_attendance()
                event.accept()
            elif reply == QMessageBox.No:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set application properties
    app.setApplicationName("Attendance Manager")
    app.setApplicationVersion("1.0")
    
    window = AttendanceGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()