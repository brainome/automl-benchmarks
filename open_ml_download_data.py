"""
this script is used to download TRAIN_URL into cached TRAIN_FILE
"""

import csv
import urllib.request
from pathlib import Path

SUITE_FILE = './test-suites/open_ml_select.tsv'

with open(SUITE_FILE, newline='') as scenario:
    reader = csv.DictReader(scenario, delimiter='\t')
    for idx, params in enumerate(reader):
        train_filename = params['TRAIN_FILE']
        train_url = params['TRAIN_URL']
        train_file = Path(train_filename)
        if not train_file.exists():
            uri_filename = urllib.request.\
                urlopen(urllib.request.Request(train_url, method='HEAD')).\
                info().get_filename()
            url_file = train_file.parents[0] / uri_filename
            print(f'{train_filename} does not exist, downloading {url_file}')
            urllib.request.urlretrieve(train_url, url_file)
