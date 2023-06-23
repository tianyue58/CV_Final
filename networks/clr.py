# https://github.com/beresandras/contrastive-classification-keras

import tensorflow as tf

from keras import Model
from keras.models import clone_model
from keras.activations import softmax
from keras.metrics import SparseCategoricalAccuracy
from keras.losses import categorical_crossentropy
from keras.losses import SparseCategoricalCrossentropy
from keras.losses import sparse_categorical_crossentropy
from abc import abstractmethod


class ContrastiveModel(Model):
    def __init__(self, contrastive_augmenter, classification_augmenter, encoder, projection_head, linear_probe):
        super().__init__()
        self.probe_accuracy = None
        self.correlation_accuracy = None
        self.contrastive_accuracy = None
        self.probe_loss = None
        self.probe_optimizer = None
        self.contrastive_optimizer = None
        self.contrastive_augmenter = contrastive_augmenter
        self.classification_augmenter = classification_augmenter
        self.encoder = encoder
        self.projection_head = projection_head
        self.linear_probe = linear_probe

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

        # self.contrastive_loss will be defined as a method
        self.probe_loss = SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_accuracy = SparseCategoricalAccuracy()
        self.correlation_accuracy = SparseCategoricalAccuracy()
        self.probe_accuracy = SparseCategoricalAccuracy()

    def reset_metrics(self):
        self.contrastive_accuracy.reset_states()
        self.correlation_accuracy.reset_states()
        self.probe_accuracy.reset_states()

    def update_contrastive_accuracy(self, features_1, features_2):
        # self-supervised metric inspired by the SimCLR loss

        # cosine similarity: the dot product of the l2-normalized feature vectors
        features_1 = tf.math.l2_normalize(features_1, axis=1)
        features_2 = tf.math.l2_normalize(features_2, axis=1)
        similarities = tf.matmul(features_1, features_2, transpose_b=True)

        # the similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(features_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0))

    def update_correlation_accuracy(self, features_1, features_2):
        # self-supervised metric inspired by the BarlowTwins loss

        # normalization so that cross-correlation will be between -1 and 1
        features_1 = (features_1 - tf.reduce_mean(features_1, axis=0)) / tf.math.reduce_std(features_1, axis=0)
        features_2 = (features_2 - tf.reduce_mean(features_2, axis=0)) / tf.math.reduce_std(features_2, axis=0)

        # the cross correlation of image representations should be the identity matrix
        batch_size = tf.shape(features_1, out_type=tf.float32)[0]
        cross_correlation = (tf.matmul(features_1, features_2, transpose_a=True) / batch_size)

        feature_dim = tf.shape(features_1)[1]
        correlation_labels = tf.range(feature_dim)
        self.correlation_accuracy.update_state(
            tf.concat([correlation_labels, correlation_labels], axis=0),
            tf.concat([cross_correlation, tf.transpose(cross_correlation)], axis=0))

    @abstractmethod
    def contrastive_loss(self, projections_1, projections_2):
        pass

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data

        # both labeled and unlabeled images are used, without labels
        images = tf.concat((unlabeled_images, labeled_images), axis=0)
        # each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            # the representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(contrastive_loss,
                                  self.encoder.trainable_weights + self.projection_head.trainable_weights)
        self.contrastive_optimizer.apply_gradients(
            zip(gradients, self.encoder.trainable_weights + self.projection_head.trainable_weights))
        self.update_contrastive_accuracy(features_1, features_2)
        self.update_correlation_accuracy(features_1, features_2)

        # labels are only used in evaluation for an on-the-fly logistic regression
        preprocessed_images = self.classification_augmenter(labeled_images)
        with tf.GradientTape() as tape:
            features = self.encoder(preprocessed_images)
            class_logits = self.linear_probe(features)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(zip(gradients, self.linear_probe.trainable_weights))
        self.probe_accuracy.update_state(labels, class_logits)

        return {
            "c_loss": contrastive_loss,
            "c_acc": self.contrastive_accuracy.result(),
            "r_acc": self.correlation_accuracy.result(),
            "p_loss": probe_loss,
            "p_acc": self.probe_accuracy.result()
        }

    def test_step(self, data):
        labeled_images, labels = data

        preprocessed_images = self.classification_augmenter(labeled_images, training=False)
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)

        self.probe_accuracy.update_state(labels, class_logits)
        return {"p_loss": probe_loss, "p_acc": self.probe_accuracy.result()}


class MomentumContrastiveModel(ContrastiveModel):
    def __init__(self, contrastive_augmenter, classification_augmenter, encoder,
                 projection_head, linear_probe, momentum_coeff,):
        super().__init__(contrastive_augmenter, classification_augmenter, encoder, projection_head, linear_probe)
        self.momentum_coeff = momentum_coeff

        # the momentum networks are initialized from their online counterparts
        self.m_encoder = clone_model(self.encoder)
        self.m_projection_head = clone_model(self.projection_head)

    @abstractmethod
    def contrastive_loss(self, projections_1, projections_2, m_projections_1, m_projections_2,):
        pass

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data

        images = tf.concat((unlabeled_images, labeled_images), axis=0)
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            m_features_1 = self.m_encoder(augmented_images_1)
            m_features_2 = self.m_encoder(augmented_images_2)
            m_projections_1 = self.m_projection_head(m_features_1)
            m_projections_2 = self.m_projection_head(m_features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2, m_projections_1, m_projections_2)
        gradients = tape.gradient(contrastive_loss,
                                  self.encoder.trainable_weights + self.projection_head.trainable_weights)
        self.contrastive_optimizer.apply_gradients(
            zip(gradients, self.encoder.trainable_weights + self.projection_head.trainable_weights))
        self.update_contrastive_accuracy(m_features_1, m_features_2)
        self.update_correlation_accuracy(m_features_1, m_features_2)

        preprocessed_images = self.classification_augmenter(labeled_images)
        with tf.GradientTape() as tape:
            # the momentum encoder is used here as it moves more slowly
            features = self.m_encoder(preprocessed_images)
            class_logits = self.linear_probe(features)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(zip(gradients, self.linear_probe.trainable_weights))
        self.probe_accuracy.update_state(labels, class_logits)

        # the momentum networks are updated by exponential moving average
        for weight, m_weight in zip(self.encoder.weights, self.m_encoder.weights):
            m_weight.assign(self.momentum_coeff * m_weight + (1 - self.momentum_coeff) * weight)
        for weight, m_weight in zip(self.projection_head.weights, self.m_projection_head.weights):
            m_weight.assign(self.momentum_coeff * m_weight + (1 - self.momentum_coeff) * weight)

        return {
            "c_loss": contrastive_loss,
            "c_acc": self.contrastive_accuracy.result(),
            "r_acc": self.correlation_accuracy.result(),
            "p_loss": probe_loss,
            "p_acc": self.probe_accuracy.result()
        }


class SimCLR(ContrastiveModel):
    def __init__(self, contrastive_augmenter, classification_augmenter, encoder, projection_head,
                 linear_probe, temperature):
        super().__init__(contrastive_augmenter, classification_augmenter, encoder, projection_head, linear_probe)
        self.temperature = temperature

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature)

        # the temperature-scaled similarities are used as logits for cross-entropy
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = sparse_categorical_crossentropy(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0),
            from_logits=True)
        return loss


class NNCLR(ContrastiveModel):
    def __init__(self, contrastive_augmenter, classification_augmenter, encoder, projection_head, linear_probe,
                 temperature, queue_size):
        super().__init__(contrastive_augmenter, classification_augmenter, encoder, projection_head, linear_probe)
        self.temperature = temperature

        feature_dimensions = encoder.output_shape[1]
        self.feature_queue = tf.Variable(
            tf.math.l2_normalize(tf.random.normal(shape=(queue_size, feature_dimensions)), axis=1),
            trainable=False)

    def nearest_neighbour(self, projections):
        # highest cosine similarity == lowest L2 distance, for L2 normalized features
        support_similarities = tf.matmul(projections, self.feature_queue, transpose_b=True)

        # hard nearest-neighbours
        nn_projections = tf.gather(self.feature_queue, tf.argmax(support_similarities, axis=1), axis=0)

        # straight-through gradient estimation
        # paper used stop gradient, however it helps performance at this scale
        return projections + tf.stop_gradient(nn_projections - projections)

    def contrastive_loss(self, projections_1, projections_2):
        # similar to the SimCLR loss, however we take the nearest neighbours of a set
        # of projections from a feature queue
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities_1_2 = (tf.matmul(self.nearest_neighbour(projections_1), projections_2, transpose_b=True) / self.temperature)
        similarities_2_1 = (tf.matmul(self.nearest_neighbour(projections_2), projections_1, transpose_b=True) / self.temperature)

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = sparse_categorical_crossentropy(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities_1_2, similarities_2_1], axis=0),
            from_logits=True)

        # feature queue update
        self.feature_queue.assign(tf.concat([projections_1, self.feature_queue[:-batch_size]], axis=0))
        return loss


class DCCLR(ContrastiveModel):
    def __init__(self, contrastive_augmenter, classification_augmenter, encoder, projection_head,
                 linear_probe, temperature):
        super().__init__(contrastive_augmenter, classification_augmenter, encoder, projection_head, linear_probe)
        self.temperature = temperature

    def contrastive_loss(self, projections_1, projections_2):
        # a modified InfoNCE loss, which should provide better performance at
        # lower batch sizes

        # cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature)

        # the similarities of the positives (the main diagonal) are masked and
        # are not included in the softmax normalization
        batch_size = tf.shape(projections_1)[0]
        decoupling_mask = 1.0 - tf.eye(batch_size)
        decoupled_similarities = decoupling_mask * tf.exp(similarities)

        loss = tf.reduce_mean(
            -tf.linalg.diag_part(similarities) + tf.math.log(
                tf.reduce_sum(decoupled_similarities, axis=0) + tf.reduce_sum(decoupled_similarities, axis=1)))
        # the sum along the two axes should be put in separate log-sum-exp
        # expressions according to the paper, this however achieves slightly
        # higher performance at this scale

        return loss


class BarlowTwins(ContrastiveModel):
    def __init__(self, contrastive_augmenter, classification_augmenter, encoder, projection_head,
                 linear_probe, redundancy_reduction_weight):
        super().__init__(contrastive_augmenter, classification_augmenter, encoder, projection_head, linear_probe)
        # weighting coefficient between the two loss components
        self.redundancy_reduction_weight = redundancy_reduction_weight
        # its value differs from the paper, because the loss implementation has been
        # changed to be invariant to the encoder output dimensions (feature dim)

    def contrastive_loss(self, projections_1, projections_2):
        projections_1 = (projections_1 - tf.reduce_mean(projections_1, axis=0)) / tf.math.reduce_std(projections_1, axis=0)
        projections_2 = (projections_2 - tf.reduce_mean(projections_2, axis=0)) / tf.math.reduce_std(projections_2, axis=0)

        # the cross correlation of image representations should be the identity matrix
        batch_size = tf.shape(projections_1, out_type=tf.float32)[0]
        feature_dim = tf.shape(projections_1, out_type=tf.float32)[1]
        cross_correlation = (tf.matmul(projections_1, projections_2, transpose_a=True) / batch_size)
        target_cross_correlation = tf.eye(feature_dim)
        squared_errors = (target_cross_correlation - cross_correlation) ** 2

        # invariance loss = average diagonal error
        # redundancy reduction loss = average off-diagonal error
        invariance_loss = (tf.reduce_sum(squared_errors * tf.eye(feature_dim)) / feature_dim)
        redundancy_reduction_loss = tf.reduce_sum(
            squared_errors * (1 - tf.eye(feature_dim))) / (feature_dim * (feature_dim - 1))
        return invariance_loss + self.redundancy_reduction_weight * redundancy_reduction_loss


class HSICTwins(ContrastiveModel):
    def __init__(self, contrastive_augmenter, classification_augmenter, encoder, projection_head,
                 linear_probe, redundancy_reduction_weight):
        super().__init__(contrastive_augmenter, classification_augmenter, encoder, projection_head, linear_probe)
        # weighting coefficient between the two loss components
        self.redundancy_reduction_weight = redundancy_reduction_weight
        # its value differs from the paper, because the loss implementation has been
        # changed to be invariant to the encoder output dimensions (feature dim)

    def contrastive_loss(self, projections_1, projections_2):
        # a modified BarlowTwins loss, derived from Hilbert-Schmidt Independence
        # Criterion maximization, the only difference is the target cross correlation

        projections_1 = (projections_1 - tf.reduce_mean(
            projections_1, axis=0)) / tf.math.reduce_std(projections_1, axis=0)
        projections_2 = (projections_2 - tf.reduce_mean(
            projections_2, axis=0)) / tf.math.reduce_std(projections_2, axis=0)

        # the cross correlation of image representations should be 1 along the diagonal
        # and -1 everywhere else
        batch_size = tf.shape(projections_1, out_type=tf.float32)[0]
        feature_dim = tf.shape(projections_1, out_type=tf.float32)[1]
        cross_correlation = (tf.matmul(projections_1, projections_2, transpose_a=True) / batch_size)
        target_cross_correlation = 2.0 * tf.eye(feature_dim) - 1.0
        squared_errors = (target_cross_correlation - cross_correlation) ** 2

        # invariance loss = average diagonal error
        # redundancy reduction loss = average off-diagonal error
        invariance_loss = (tf.reduce_sum(squared_errors * tf.eye(feature_dim)) / feature_dim)
        redundancy_reduction_loss = tf.reduce_sum(
            squared_errors * (1 - tf.eye(feature_dim))) / (feature_dim * (feature_dim - 1))
        return invariance_loss + self.redundancy_reduction_weight * redundancy_reduction_loss


class TWIST(ContrastiveModel):
    def __init__(self, contrastive_augmenter, classification_augmenter, encoder, projection_head, linear_probe):
        super().__init__(contrastive_augmenter, classification_augmenter, encoder, projection_head, linear_probe)

    def contrastive_loss(self, projections_1, projections_2):
        # a probabilistic, hyperparameter- and negative-free loss

        # batch normalization before softmax operation
        projections_1 = (projections_1 - tf.reduce_mean(
            projections_1, axis=0)) / tf.math.reduce_std(projections_1, axis=0)
        projections_2 = (projections_2 - tf.reduce_mean(
            projections_2, axis=0)) / tf.math.reduce_std(projections_2, axis=0)

        probabilities_1 = softmax(projections_1)
        probabilities_2 = softmax(projections_2)

        mean_probabilities_1 = tf.reduce_mean(probabilities_1, axis=0)
        mean_probabilities_2 = tf.reduce_mean(probabilities_2, axis=0)

        # cross-entropy(1,2): KL-div(1,2) (consistency) + entropy(1) (sharpness)
        # -cross-entropy(mean1,mean1): -entropy(mean1) (diversity)
        loss = categorical_crossentropy(
            tf.concat([probabilities_1, probabilities_2], axis=0),
            tf.concat([probabilities_2, probabilities_1], axis=0),) - categorical_crossentropy(
            tf.concat([mean_probabilities_1, mean_probabilities_2], axis=0), tf.concat(
                [mean_probabilities_1, mean_probabilities_2], axis=0))
        return loss


class MoCo(MomentumContrastiveModel):
    def __init__(self, contrastive_augmenter, classification_augmenter, encoder, projection_head,
                 linear_probe, momentum_coeff, temperature, queue_size):
        super().__init__(contrastive_augmenter, classification_augmenter, encoder, projection_head,
                         linear_probe, momentum_coeff)
        self.temperature = temperature
        feature_dimensions = encoder.output_shape[1]
        self.feature_queue = tf.Variable(tf.math.l2_normalize(
            tf.random.normal(shape=(queue_size, feature_dimensions)), axis=1), trainable=False)

    def contrastive_loss(self, projections_1, projections_2, m_projections_1, m_projections_2,):
        # similar to the SimCLR loss, however it uses the momentum networks'
        # representations of the differently augmented views as targets
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        m_projections_1 = tf.math.l2_normalize(m_projections_1, axis=1)
        m_projections_2 = tf.math.l2_normalize(m_projections_2, axis=1)

        similarities_1_2 = (tf.matmul(projections_1, tf.concat((
            m_projections_2, self.feature_queue), axis=0), transpose_b=True,) / self.temperature)
        similarities_2_1 = (tf.matmul(projections_2, tf.concat((
            m_projections_1, self.feature_queue), axis=0), transpose_b=True,) / self.temperature)

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = sparse_categorical_crossentropy(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities_1_2, similarities_2_1], axis=0),
            from_logits=True)

        # feature queue update
        self.feature_queue.assign(tf.concat([m_projections_1, m_projections_2,
                                             self.feature_queue[: -(2 * batch_size)],], axis=0,))
        return loss


class DINO(MomentumContrastiveModel):
    def __init__(self, contrastive_augmenter, classification_augmenter, encoder, projection_head,
                 linear_probe, momentum_coeff, temperature, sharpening):
        super().__init__(contrastive_augmenter, classification_augmenter, encoder, projection_head,
                         linear_probe, momentum_coeff)
        self.temperature = temperature
        self.sharpening = sharpening

    def contrastive_loss(self, projections_1, projections_2, m_projections_1, m_projections_2):
        # this loss does not use any negatives, needs centering + sharpening + momentum
        # to avoid collapse

        # l2-normalization is part of the projection head in the original implementation
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        m_projections_1 = tf.math.l2_normalize(m_projections_1, axis=1)
        m_projections_2 = tf.math.l2_normalize(m_projections_2, axis=1)

        center = tf.reduce_mean(tf.concat([m_projections_1, m_projections_2], axis=0), axis=0, keepdims=True)
        target_probabilities_1 = softmax((m_projections_1 - center) / (self.sharpening * self.temperature))
        target_probabilities_2 = softmax((m_projections_2 - center) / (self.sharpening * self.temperature))

        pred_probabilities_1 = softmax(projections_1 / self.temperature)
        pred_probabilities_2 = softmax(projections_2 / self.temperature)

        loss = categorical_crossentropy(
            tf.concat([target_probabilities_1, target_probabilities_2], axis=0),
            tf.concat([pred_probabilities_2, pred_probabilities_1], axis=0))
        return loss
