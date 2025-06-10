import psycopg2
from datetime import datetime
from config import CONFIG

class PostgresLogger:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=CONFIG["postgres"]["host"],
            database=CONFIG["postgres"]["database"],
            user=CONFIG["postgres"]["user"],
            password=CONFIG["postgres"]["password"],
            port=CONFIG["postgres"]["port"]
        )
        self.cursor = self.conn.cursor()
    
    def log_detection(self, person_name, camera_ip, classroom):
        now = datetime.now()
        self.cursor.execute(
            "INSERT INTO faculty_log (person_name, camera_ip, classroom, timestamp) VALUES (%s, %s, %s, %s)",
            (person_name, camera_ip, classroom, now)
        )
        self.conn.commit()
    
    def close(self):
        self.cursor.close()
        self.conn.close()
