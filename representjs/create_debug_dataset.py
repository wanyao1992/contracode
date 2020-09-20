import gzip
import pickle
import os
data_dir = '/home/wanyao/Dropbox/ghproj-titan/contracode/data/codesearchnet_javascript'
full_path = '/home/wanyao/Dropbox/ghproj-titan/contracode/data/codesearchnet_javascript/javascript_augmented.pickle.gz'

with gzip.open(str(full_path), "rb") as f, open(os.path.join(data_dir, 'javascript_augmented_debug.pickle'), 'wb') as f_debug:
    examples = pickle.load(f)
    examples_debug = examples[:100]
    pickle.dump(examples_debug, f_debug)