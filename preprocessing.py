import ultralytics
from ultralytics.data.split import autosplit

autosplit(
    path='/Users/grace/msc_project/Training_code/data_2/images/',
    weights=(0.8, 0.1, 0.1),  # (train, validation, test) fractional splits
    annotated_only=True,  # split only images with annotation file when True
)