import os

CONFIG = {
    "training_data_path": "/Users/akashzamnani/Downloads/Attendance Management System 24-25",
    'labels_file': 'data/comp_faculty_labels.pkl',
    "data_file": "faiss_index.idx",
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
            "hikvision_ip": "10.10.123.140",
            "hikvision_port": 554,
            "classroom": "510",
            "hikvision_user": "DYPDPU",
            "hikvision_password": "Admin@123",
            "hikvision_stream": "Streaming/Channels/101",
        },
        {
            "hikvision_ip": "10.10.123.140",
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
        # "host": "ep-curly-dust-a157hm5c-pooler.ap-southeast-1.aws.neon.tech",
        "host":"localhost",
        "database": "neondb",
        "user": "neondb_owner",
        "password": "npg_Ambc9E7WGtnM",
        "port": 5432
    }
}
link='postgresql://neondb_owner:npg_Ambc9E7WGtnM@ep-curly-dust-a157hm5c-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require'