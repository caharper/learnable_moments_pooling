import tensorflow as tf
from learnable_stats.layers.learnable_stats.stats import (
    central_moment,
    standardized_moment,
)


class StatLogger(tf.keras.callbacks.Callback):
    def __init__(self, eval_ds, layer_name, **kwargs):
        super().__init__(**kwargs)
        self.eval_ds = eval_ds
        self.layer_name = layer_name

        self.means = []
        self.variances = []
        self.skews = []
        self.kurtosises = []

    def _get_chopped_model(self):
        return tf.keras.models.Model(
            self.model.inputs, self.model.get_layer(self.layer_name).output
        )

    def _get_stats(self):
        chopped_model = self._get_chopped_model()

        means = []
        variances = []
        skews = []
        kurtosises = []

        for d in self.eval_ds:
            preds = chopped_model.predict(d)
            m = tf.reduce_mean(preds, axis=1)
            v = central_moment(preds, 2.0)
            s = standardized_moment(preds, 3.0)
            k = standardized_moment(preds, 4.0)

            m = tf.reduce_mean(m)
            v = tf.reduce_mean(v)
            s = tf.reduce_mean(s)
            k = tf.reduce_mean(k)

            means.append(m)
            variances.append(v)
            skews.append(s)
            kurtosises.append(k)

        return (
            tf.reduce_mean(means),
            tf.reduce_mean(variances),
            tf.reduce_mean(skews),
            tf.reduce_mean(kurtosises),
        )

    def _add_stats(self):
        m, v, s, k = self._get_stats()
        self.means.append(m)
        self.variances.append(v)
        self.skews.append(s)
        self.kurtosises.append(k)

    def on_train_begin(self, logs=None):
        self._add_stats()

    def on_epoch_end(self, epoch, logs=None):
        self._add_stats()
