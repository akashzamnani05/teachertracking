import pickle
import os
import faiss

class DataHandler:
    def __init__(self, data_file, labels_file):
        self.labels_file = labels_file
        self.data_file = data_file
        self.known_embeddings = []
        self.known_labels = []

    def load_data(self):
        if os.path.exists(self.data_file):
            try:
                self.known_embeddings =  faiss.read_index('faiss_index.idx')
                
                print(f"Loaded {len(self.known_labels)} known faces")
            except Exception as e:
                print(f"Error loading face data: {e}")
                self.known_embeddings = []
                self.known_labels = []
        else:
            print(f"No face embeddings data file found at {self.data_file}")

    def load_labels(self):
        if os.path.exists(self.labels_file):
            try:
                with open(self.labels_file, 'rb') as f:
                    data = pickle.load(f)
                    # self.known_embeddings = data['embeddings']
                    self.known_labels = data['labels']
                    print(f"Loaded {len(self.known_labels)} known faces labels")
            except Exception as e:
                print(f"Error loading face data: {e}")
                self.known_embeddings = []
                self.known_labels = []
        else:
            print(f"No face labels data file found at {self.labels_file}")


    def save_data(self):
        try:
            with open(self.labels_file, 'wb') as f:
                pickle.dump({
                    'labels': self.known_labels
                }, f)
            faiss.write_index(self.known_embeddings, self.data_file)
        except Exception as e:
            print(f"Error saving data: {e}")