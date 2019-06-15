import tensorflow as tf
from typing import Any

def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims


def is_tf_expression(x: Any) -> bool:
    """Check whether the input is a valid Tensorflow expression, i.e., Tensorflow Tensor, Variable, or Operation."""
    return isinstance(x, (tf.Tensor, tf.Variable, tf.Operation))