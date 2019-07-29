import tensorflow as tf


def sigmoid(labels, train_logits):
    return tf.reduce_mean(
        # loss something
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, dtype=tf.float32), logits=train_logits)
    )

def sparse_softmax(labels, endings):
    return tf.reduce_mean(
            # loss something
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=endings)
    )