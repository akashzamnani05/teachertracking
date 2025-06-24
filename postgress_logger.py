import psycopg2
from datetime import datetime
from config import CONFIG
import logging

class PostgresLogger:
    def __init__(self):
        try:
            self.conn = psycopg2.connect(
                host=CONFIG["postgres"]["host"],
                database=CONFIG["postgres"]["database"],
                user=CONFIG["postgres"]["user"],
                password=CONFIG["postgres"]["password"],
                port=CONFIG["postgres"]["port"]
            )
            self.conn.autocommit = False  # Explicit transaction management
            self._create_table_if_not_exists()
            print("PostgreSQL connection established successfully")
        except Exception as e:
            print(f"Failed to connect to PostgreSQL: {e}")
            logging.error(f"PostgreSQL connection error: {e}")
            self.conn = None
    
    def _create_table_if_not_exists(self):
        if not self.conn:
            return
            
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'faculty_logs'
                """)
                columns = {row[0] for row in cur.fetchall()}
                required_columns = {'id', 'person_name','erp_id' ,'camera_ip', 'classroom', 'timestamp', 'created_at'}
                
                if not columns.issuperset(required_columns):
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS faculty_logs (
                            id SERIAL PRIMARY KEY,
                            person_name VARCHAR(255) NOT NULL,
                            erp_id VARCHAR(255) NOT NULL,
                            camera_ip VARCHAR(45) NOT NULL,
                            classroom VARCHAR(255) NOT NULL,
                            timestamp TIMESTAMP NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_faculty_log_timestamp 
                    ON faculty_logs(timestamp);
                """)
            self.conn.commit()
            print("Faculty log table created/verified successfully")
        except Exception as e:
            self.conn.rollback()
            print(f"Error creating table: {e}")
            logging.error(f"PostgreSQL table creation error: {e}")
    
    def log_detection(self, person_name, erp_id, camera_ip, classroom):
        if not self.conn or self.conn.closed:
            print("PostgreSQL connection not available")
            return False
            
        # CHANGE: Added validation for erp_id parameter
        if not all([person_name, erp_id, camera_ip, classroom]):
            print("Invalid input: All fields must be non-empty")
            logging.error("Invalid input: person_name, erp_id, camera_ip, or classroom is empty")
            return False
            
        # CHANGE: Added length validation for erp_id
        if len(camera_ip) > 45 or len(person_name) > 255 or len(classroom) > 255 or len(erp_id) > 255:
            print("Invalid input: Field length exceeds column limits")
            logging.error("Input exceeds column length limits")
            return False
            
        try:
            now = datetime.now()
            try:
                with self.conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO faculty_logs (person_name, erp_id, camera_ip, classroom, timestamp) 
                        VALUES (%s, %s, %s, %s, %s)""",
                        (person_name, erp_id, camera_ip, classroom, now)
                    )
            except Exception as e:
                print(e)

            
            self.conn.commit()
            print(f"Logged detection: {person_name} (ERP: {erp_id}) at {camera_ip} in {classroom}")
            return True
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"PostgreSQL error during logging: {e}")
            logging.error(f"PostgreSQL logging error: {e}")
            return False
        except Exception as e:
            self.conn.rollback()
            print(f"Unexpected error during logging: {e}")
            logging.error(f"Unexpected logging error: {e}")
            return False
    
    def get_recent_detections(self, hours=24):
        if not self.conn or self.conn.closed:
            return []
            
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """SELECT person_name, erp_id, camera_ip, classroom, timestamp 
                       FROM faculty_logs 
                       WHERE timestamp >= NOW() - %s::interval
                       ORDER BY timestamp DESC""",
                    (f"{hours} hours",)
                )
                return cur.fetchall()
        except Exception as e:
            print(f"Error fetching recent detections: {e}")
            logging.error(f"Error fetching recent detections: {e}")
            return []
    
    def test_connection(self):
        if not self.conn or self.conn.closed:
            return False
            
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            logging.error(f"Connection test failed: {e}")
            return False
    
    def close(self):
        try:
            if self.conn and not self.conn.closed:
                self.conn.close()
            print("PostgreSQL connection closed")
        except Exception as e:
            print(f"Error closing PostgreSQL connection: {e}")
            logging.error(f"Error closing PostgreSQL connection: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()