"""Module for running the training of the machine learning model.

Scripts that performs all the steps to train the ML model.
"""
import logging
import argparse
import time
import tensorflow as tf

from helpers import preprocess, models, metrics
from run_preprocess import run_preprocess
from run_train import run_training


def run_preprocess_and_train(args):
  """Runs the ML model pipeline.

  Args:
    args: args that are passed when submitting the training

  Returns:

  """
  run_preprocess(args)
  # TODO: pass argument from prepro to train instead of having a ful list of args
  run_training(args)


def main():
  parser = argparse.ArgumentParser(description='Preprocess and Train')

  parser.add_argument('--cutoff_year',
                      type=str,
                      help='Cutoff year for the stock data',
                      default='2010')

  parser.add_argument('--data_bucket',
                      type=str,
                      help='GCS bucket where preprocessed data is saved',
                      default='<your-bucket-name>')

  parser.add_argument('--blob_path',
                      type=str,
                      help='GCS blob path where data is saved',
                      default='<your-bucket-name>')

  parser.add_argument('--model',
                      type=str,
                      help='model to be used for training',
                      default='DeepModel',
                      choices=['FlatModel', 'DeepModel'])

  parser.add_argument('--epochs',
                      type=int,
                      help='number of epochs to train',
                      default=30001)

  parser.add_argument('--version',
                      type=str,
                      help='version (stored for serving)',
                      default='1')

  parser.add_argument('--model_bucket',
                      type=str,
                      help='GCS bucket where model is saved',
                      default='<your-bucket-name>')

  args = parser.parse_args()
  run_preprocess_and_train(args)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  main()
