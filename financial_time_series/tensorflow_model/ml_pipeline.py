#!/usr/bin/env python3
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import kfp.dsl as dsl


class Preprocess(dsl.ContainerOp):

  def __init__(self, name, data_bucket, cutoff_year):
    super(Preprocess, self).__init__(
      name=name,
      # image needs to be a compile-time string
      image='gcr.io/sven-sandbox/kubeflow/cpu:v3',
      command=['python3', 'run_preprocess.py'],
      arguments=[
        '--data_bucket', data_bucket,
        '--cutoff_year', cutoff_year
      ],
      file_outputs={'blob-path': 'data/data_stocks.csv'}
    )

class Train(dsl.ContainerOp):

  def __init__(self, name, blob_path, version, data_bucket, model_bucket):
    super(Train, self).__init__(
      name=name,
      # image needs to be a compile-time string
      image='gcr.io/sven-sandbox/kubeflow/cpu:v3',
      command=['python3', 'run_train.py'],
      arguments=[
        '--version', version,
        '--blob_path', blob_path,
        '--data_bucket', data_bucket,
        '--model_bucket', model_bucket
      ]
    )


@dsl.pipeline(
  name='financial time series',
  description='Train Financial Time Series'
)
def train_and_deploy(
        data_bucket=dsl.PipelineParam('data-bucket', value='kf-data'),
        cutoff_year=dsl.PipelineParam('cutoff-year', value='2010'),
        model_bucket=dsl.PipelineParam('model-bucket', value='kf-finance'),
        version=dsl.PipelineParam('version', value='8')
):
  """Pipeline to train financial time series model"""
  preprocess_op = Preprocess('preprocess', data_bucket, cutoff_year)
  train_op = Train('train and deploy', preprocess_op.output, version, data_bucket, model_bucket)


if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(train_and_deploy, __file__ + '.tar.gz')