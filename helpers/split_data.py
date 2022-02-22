import os
import sys
import os
import argparse
import numpy as np
import subprocess
import csv
from sklearn.model_selection import StratifiedShuffleSplit


def remove_clean_CSVs():
	clean_CSVs = [f for f in os.listdir('.') if os.path.isfile(f) and '.csv' in f and 'clean' in f]
	for CSV in clean_CSVs:
		os.remove(CSV)
	if os.path.exists('clean.state'):
		os.remove('clean.state')


def clean_CSV_with_btc(CSV, dest, target):
	remove_clean_CSVs()
	cmd = f'brainome {CSV} -cleanonly'
	if target:
		cmd += f' -target {target}'
	subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	fname = [f for f in os.listdir('.') if os.path.isfile(f) and '.csv' in f and 'clean' in f][0]
	new_name = CSV.split(os.sep)[-1].replace('.csv', '-clean.csv')
	path_to_clean_csv = f'{dest}{os.sep}{new_name}'
	os.rename(fname, path_to_clean_csv)
	return path_to_clean_csv


def get_header(CSV, target):
	with open(CSV) as f:
		reader = csv.reader(f)
		header = next(reader)
	if target:
		header.remove(target)
		header.append(target)
	# google has restrictions on header names
	header = [column_name.replace('-', '_').replace('.', '_').replace('?', '').replace(',', '_').replace(' ', '_') for column_name in header[:-1]] + [header[-1]]
	return np.array(header).reshape(1, -1)


def split(CSV, dest, target=''):

	path_to_clean_csv = clean_CSV_with_btc(CSV, dest, target=target)
	arr = np.loadtxt(path_to_clean_csv, delimiter=',', dtype=str)
	X, y = arr[:, :-1], arr[:, -1]
	n_classes = np.unique(y.reshape(-1)).shape[0]
	sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)

	split = next(sss.split(X, y))
	idxs_train, idxs_test = split[0], split[1]

	X_train, X_test = X[idxs_train], X[idxs_test]
	y_train, y_test = y[idxs_train], y[idxs_test]

	arr_train = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
	arr_test = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)

	header = get_header(CSV, target)

	arr_train_headered = np.concatenate((header, arr_train))
	arr_test_headered = np.concatenate((header, arr_test))

	outfile_train = path_to_clean_csv.replace('.csv', '-train.csv')
	outfile_test = path_to_clean_csv.replace('.csv', '-test.csv')
	outfile_test_targetless = path_to_clean_csv.replace('.csv', '-test-targetless.csv')

	np.savetxt(outfile_train, arr_train_headered, delimiter=',', fmt='%s')
	np.savetxt(outfile_test, arr_test_headered, delimiter=',', fmt='%s')
	np.savetxt(outfile_test_targetless, X_test, delimiter=',', fmt='%s')

	arr_test_headered_targetless = arr_test_headered[:, :-1]
	np.savetxt(outfile_test_targetless.replace('.csv', '-headered.csv'), arr_test_headered_targetless, delimiter=',', fmt='%s')

	return n_classes

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('CSV')
	parser.add_argument('dest')
	parser.add_argument('-target', type=str, default='')
	args = parser.parse_args()
	split(args.CSV, args.dest, args.target)