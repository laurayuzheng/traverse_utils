from datasets.persona_dataset import PersonaDataset

class BasicDataset(PersonaDataset):

    def __init__(self, config=None, is_validation=False):
        if is_validation:
            self.data_path = config['val_data_path']
        else:
            self.data_path = config['train_data_path']
        self.is_validation = is_validation
        self.config = config
        self.data_loaded_memory = []
        self.file_cache = {}

        self.config['store_data_in_memory'] = True 

        self.starting_frame = self.config['starting_frame'][0]