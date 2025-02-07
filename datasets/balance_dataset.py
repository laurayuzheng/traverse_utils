from unitraj.datasets.base_dataset import BaseDataset
import numpy as np

class BalancedDataset(BaseDataset):
    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)
        self.balance_data()

    def balance_data(self):
        # First, we need the trajectory type of each sample.
        # If data is stored in memory, we can directly access it.
        # If not, we'll have to load each sample once to get the trajectory_type.
        
        if self.config.get('store_data_in_memory', False):
            trajectory_types = []
            for data in self.data_loaded_memory:
                if 'trajectory_type' not in data:
                    raise KeyError("trajectory_type not found in data. Ensure postprocess is producing it.")
                trajectory_types.append(data['trajectory_type'])
        else:
            trajectory_types = []
            for key in self.data_loaded_keys:
                record = self._get_record(key)
                if 'trajectory_type' not in record:
                    raise KeyError("trajectory_type not found in record. Ensure postprocess is producing it.")
                trajectory_types.append(record['trajectory_type'])

        # Count occurrences of each class
        from collections import Counter
        class_counts = Counter(trajectory_types)

        # Decide on balancing strategy. Let's oversample/undersample to the portion of max frequency.
        max_count = int(max(class_counts.values()) // (1 / self.config.portion))

        # Create a mapping from class to a list of indices
        class_to_indices = {}
        for i, ctype in enumerate(trajectory_types):
            if ctype not in class_to_indices:
                class_to_indices[ctype] = []
            class_to_indices[ctype].append(i)

        # Now oversample each class to have max_count samples
        self.balanced_indices = []
        for ctype, indices in class_to_indices.items():
            count = len(indices)
            if count < max_count and getattr(self.config, 'oversampling', False) :
                # Oversample by repeating
                # If you prefer a more random approach, you can do random.choices
                # or shuffle before repeating.
                oversampled = indices * (max_count // count)
                remainder = max_count % count
                if remainder > 0:
                    oversampled += indices[:remainder]
                self.balanced_indices.extend(oversampled)
            elif count > max_count and getattr(self.config, 'undersampling', False):
                undersampled = np.random.choice(indices, size=max_count, replace=False).tolist()
                self.balanced_indices.extend(undersampled)
            else:
                # If we wanted to undersample, we could slice here. For now, just add them as-is.
                self.balanced_indices.extend(indices)

        # Shuffle balanced_indices to avoid ordering effects
        np.random.shuffle(self.balanced_indices)

    def _get_record(self, key):
        # Utility function to load a single record if not in memory.
        file_info = self.data_loaded[key]
        file_path = file_info['h5_path']

        if file_path not in self.file_cache:
            self.file_cache[file_path] = self._get_file(file_path)

        group = self.file_cache[file_path][key]
        record = {k: group[k][()].decode('utf-8') if group[k].dtype.type == np.bytes_ else group[k][()] for k in group.keys()}
        return record

    def __len__(self):
        # Return length of balanced indices
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        # Retrieve the corresponding original index from balanced_indices
        balanced_idx = self.balanced_indices[idx]
        # Use the parent's logic to get the item
        if self.config.get('store_data_in_memory', False):
            # If in memory
            return self.data_loaded_memory[balanced_idx]
        else:
            # Otherwise load from file
            key = self.data_loaded_keys[balanced_idx]
            return self._get_record(key)
