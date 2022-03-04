ROOT_STATS_DIR = './experiment_data'

# Put your other constants here.
config_data = {}
config_data["dataset"] = {}
config_data['model'] = {}
config_data["dataset"]["images_root_dir"] = "data/images"
config_data['dataset']['training_ids_file_path'] = "data/train_ids.csv"
config_data['dataset']['validation_ids_file_path'] = "data/val_ids.csv"
config_data['dataset']['test_ids_file_path'] = "data/test_ids.csv"
config_data['dataset']['training_annotation_file_path'] = "data/annotations/captions_train2014.json"
config_data['dataset']['test_annotation_file_path'] = "data/annotations/captions_val2014.json"

# for baseline model
config_data['dataset']['vocabulary_threshold'] = 10
config_data['dataset']['img_size'] = (256, 256)
config_data['dataset']['batch_size'] = 20
config_data['dataset']['num_workers'] = 2
config_data['model']['hidden_size'] = 512
config_data['model']['embedding_size'] = 300
config_data['model']['model_type'] = "baseline"