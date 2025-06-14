import os

CONFIG = {
    "training_data_path": "/Users/deepakzamnani/Downloads/Attendance Management System 24-25",
    "data_file": "data/comp_faculty_new.pkl",
    "similarity_threshold": 0.4,
    "unknown_threshold": 0.2,
    "recognition_confidence": 0.2,
    "min_frames_for_recognition": 1,
    "camera_type": "hikvision",
    "cameras": [
        {
            "hikvision_ip": "10.10.122.188",
            "classroom": "502",
            "hikvision_port": 554,
            "hikvision_user": "DYPDPU",
            "hikvision_password": "Admin@123",
            "hikvision_stream": "Streaming/Channels/101",
        },
        {
            "hikvision_ip": "10.10.122.187",
            "hikvision_port": 554,
            "classroom": "505",
            "hikvision_user": "DYPDPU",
            "hikvision_password": "Admin@123",
            "hikvision_stream": "Streaming/Channels/101",
        },
        {
            "hikvision_ip": "10.10.122.189",
            "hikvision_port": 554,
            "classroom": "510",
            "hikvision_user": "DYPDPU",
            "hikvision_password": "Admin@123",
            "hikvision_stream": "Streaming/Channels/101",
        },
        {
            "hikvision_ip": "10.10.123.38",
            "hikvision_port": 554,
            "classroom": "508",
            "hikvision_user": "DYPDPU",
            "hikvision_password": "Admin@123",
            "hikvision_stream": "Streaming/Channels/101",
        }
    ],
    "face_size": (640, 640),
    "detection_scales": [1.0],
    "augmentation_factor": 15,
    "display_resolution": (1980, 1080),
    "max_clusters_per_person": 3,
    "frame_skip": 2,
    "person_detection_timeout": 60,
    "logfile": "comp.log",
    "postgres": {
        "host": "localhost",
        "database": "postgres",
        "user": "postgres",
        "password": "Akash@2003",
        "port": 5432
    }
}