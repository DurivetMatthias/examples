import kfp.dsl as dsl


class Preprocess(dsl.ContainerOp):

    def __init__(self, name, data_bucket, cutoff_year):
        super(Preprocess, self).__init__(
            name=name,
            # image needs to be a compile-time string
            image='eu.gcr.io/ml6-sandbox/cpu-matthias:latest',
            command=['python3', 'run_preprocess.py'],
            arguments=[
                '--data_bucket', data_bucket,
                '--cutoff_year', cutoff_year
            ],
            file_outputs={'blob-path': '/blob_path.txt'}
        )


class Train(dsl.ContainerOp):

    def __init__(self, name, blob_path, version, data_bucket, model_bucket):
        super(Train, self).__init__(
            name=name,
            # image needs to be a compile-time string
            image='eu.gcr.io/ml6-sandbox/cpu-matthias:latest',
            command=['python3', 'run_train.py'],
            arguments=[
                '--version', version,
                '--data_bucket', data_bucket,
                '--model_bucket', model_bucket,
                '--blob_path', blob_path
            ]
        )


@dsl.pipeline(
    name='financial time series',
    description='Train Financial Time Series'
)
def train_and_deploy(
        data_bucket=dsl.PipelineParam('data-bucket', value='kfmatthias'),
        cutoff_year=dsl.PipelineParam('cutoff-year', value='2015'),
        model_bucket=dsl.PipelineParam('model-bucket', value='kfmatthias'),
        version=dsl.PipelineParam('version', value='8')
):
    """Pipeline to train financial time series model"""
    preprocess_op = Preprocess('preprocess', data_bucket, cutoff_year)
    train_op = Train('train and deploy', preprocess_op.output,
                     version, data_bucket, model_bucket)


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(train_and_deploy, __file__ + '.tar.gz')
