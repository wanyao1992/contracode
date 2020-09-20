def get_package_root():
    import os
    from pathlib import Path
    #
    # return Path(os.path.dirname(os.path.abspath(__file__))) / ".."
    return Path('/home/wanyao/Dropbox/ghproj-titan/contracode')

PACKAGE_ROOT = get_package_root()
DATA_DIR = get_package_root() / "data"
# DATA_DIR = '/data/wanyao/ghproj_d/contracode/data'
CSNJS_DIR = DATA_DIR / "codesearchnet_javascript"
RUN_DIR = DATA_DIR / "runs"

DATA_DIR.mkdir(exist_ok=True)
RUN_DIR.mkdir(exist_ok=True)
