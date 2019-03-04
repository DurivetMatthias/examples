"""Module for running the data retrieval and preprocessing.
Scripts that performs all the steps to get the train and perform preprocessing.
"""
import argparse
import logging
import os
import shutil

from helpers import preprocess
from helpers import storage as storage_helper


def run_preprocess(args):
    """Runs the retrieval and preprocessing of the data.
    Args:
      args: args that are passed when submitting the training
    Returns:
    """
    tickers = ['snp', 'nyse', 'djia', 'nikkei',
               'hangseng', 'ftse', 'dax', 'aord']
    closing_data = preprocess.load_data(tickers, args.cutoff_year)
    time_series = preprocess.preprocess_data(closing_data)
    temp_folder = 'data'
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)
    file_path = os.path.join(
        temp_folder, 'data_{}.csv'.format(args.cutoff_year))
    time_series.to_csv(file_path, index=False)
    storage_helper.upload_to_storage(args.data_bucket, temp_folder)

    with open("/blob_path.txt", "w") as output_file:
        output_file.write(file_path)
    shutil.rmtree(temp_folder)


def main():
    parser = argparse.ArgumentParser(description='Preprocessing')

    parser.add_argument('--data_bucket',
                        type=str,
                        help='GCS bucket where preprocessed data is saved',
                        default='<your-bucket-name>')

    parser.add_argument('--cutoff_year',
                        type=str,
                        help='Cutoff year for the stock data',
                        default='2010')

    args = parser.parse_args()
    run_preprocess(args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
