import pickle
import os

class DataHandler:
    def __init__(self, data_file):
        self.data_file = data_file
        self.known_embeddings = []
        self.known_labels = []

    def load_data(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_embeddings = data['embeddings']
                    self.known_labels = data['labels']
                    print(f"Loaded {len(self.known_labels)} known faces")
            except Exception as e:
                print(f"Error loading face data: {e}")
                self.known_embeddings = []
                self.known_labels = []
        else:
            print(f"No face data file found at {self.data_file}")

    def save_data(self):
        try:
            with open(self.data_file, 'wb') as f:
                pickle.dump({
                    'embeddings': self.known_embeddings,
                    'labels': self.known_labels
                }, f)
        except Exception as e:
            print(f"Error saving data: {e}")