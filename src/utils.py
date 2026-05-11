import io
import json
import zipfile
import pandas as pd
from typing import Tuple


def read_log(zip_file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(zip_file_path, 'r') as archive:
        files = archive.namelist()
        log_filename = [f for f in files if f[-3:] == 'log']
        assert len(log_filename) == 1, "More than one log file in the log archive"
        log_filename = log_filename[0]

        with archive.open(log_filename) as log_file:
            data_log = json.load(log_file)

    activities = pd.read_csv(io.StringIO(data_log['activitiesLog']), sep=';')
    trades = pd.DataFrame(data_log['tradeHistory'])

    return trades, activities