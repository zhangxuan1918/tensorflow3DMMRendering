import tensorflow as tf

tf.profiler.experimental.client.trace('grpc://localhost:6009',
                                      '/tmp/tb_log', 2000)
