import math

import tensorflow as tf
from tqdm import tqdm

from models.CNN import CNN
from models.MLP import MLP
from utils import model_loader
from utils.data_loader import DataLoader
from models import ShuffleNetV2
from config import *


if __name__ == '__main__':
    # GPU settings
    # ==========================================================================

    gpus = tf.config.list_physical_devices(device_type='GPU')
    tf.config.set_visible_devices(devices=gpus[0], device_type='GPU')
    # for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
        # tf.config.set_logical_device_configuration(
        #     device=gpus[0],
        #     logical_devices=[tf.config.LogicalDeviceConfiguration(memory_limit=0)]
        # )

    # create model
    # ==========================================================================
    model = model_loader.load(
        mode=1,
        filepath=SAVED_MODEL_DIR+'best'
        # dirpath=SAVED_MODEL_DIR
    )

    # create dataloader
    # ==========================================================================
    train_data_loader = DataLoader(TRAIN_TFRECORD)
    valid_data_loader = DataLoader(VALID_TFRECORD)

    train_dataset = train_data_loader.get_dataset(BATCH_SIZE, augment=False)
    valid_dataset = valid_data_loader.get_dataset(BATCH_SIZE)

    total_train_num = train_data_loader.get_len()

    # create summary writer
    # ==========================================================================
    summary_writer = tf.summary.create_file_writer(SUMMARY_DIR)
    # tf.summary.trace_on(profiler=True)

    # define optimizer and loss
    # ==========================================================================

    # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-3,
    #     decay_steps=15,
    #     decay_rate=0.99,
    #     staircase=True
    # )
    # learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    #     boundaries=[500, 1500],
    #     values=[1e-3, 5e-4, 1e-4]
    # )
    learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
        # 1e-2, 3e-3, 1e-3, 3e-4,1e-4
        initial_learning_rate=1e-2,
        decay_steps=50,
        decay_rate=0.1,
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss0_obj = tf.keras.losses.SparseCategoricalCrossentropy()
    loss1_obj = tf.keras.losses.MeanSquaredError()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
    prev_valid_loss = float('inf')

    @tf.function
    def train_step(x, y0, y1):
        with tf.GradientTape() as tape:
            y0_pred, y1_pred= model(x, training=True)
            loss0 = loss0_obj(y_true=y0, y_pred=y0_pred)
            loss1 = loss1_obj(y_true=y1, y_pred=y1_pred)
            loss = loss0 + 2*loss1
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
        train_loss.update_state(values=loss)
        train_acc.update_state(y_true=y0, y_pred=y0_pred)


    @tf.function
    def valid_step(x, y0, y1):
        y0_pred, y1_pred = model(x, training=False)
        loss0 = loss0_obj(y_true=y0, y_pred=y0_pred)
        loss1 = loss1_obj(y_true=y1, y_pred=y1_pred)
        loss = loss0 + 2*loss1

        valid_loss.update_state(values=loss)
        valid_acc.update_state(y_true=y0, y_pred=y0_pred)


    # training process
    # ==========================================================================
    for epoch in range(NUM_EPOCHS):
        step = 0
        for image_batch, label_0_batch, label_1_batch in tqdm(train_dataset, desc='training'):
            train_step(image_batch, label_0_batch, label_1_batch)
            step += 1
            if step > math.ceil(total_train_num/BATCH_SIZE):
                break

        for image_batch,  label_0_batch, label_1_batch in valid_dataset:
            valid_step(image_batch, label_0_batch, label_1_batch)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch,
                                                                  NUM_EPOCHS,
                                                                  train_loss.result().numpy(),
                                                                  train_acc.result().numpy(),
                                                                  valid_loss.result().numpy(),
                                                                  valid_acc.result().numpy()))
        with summary_writer.as_default():
            tf.summary.scalar(
                name='train loss',
                data=train_loss.result().numpy(),
                step=epoch
            )
            tf.summary.scalar(
                name='train acc',
                data=train_acc.result().numpy(),
                step=epoch
            )
            tf.summary.scalar(
                name='valid loss',
                data=valid_loss.result().numpy(),
                step=epoch
            )
            tf.summary.scalar(
                name='valid acc',
                data=valid_acc.result().numpy(),
                step=epoch
            )
            tf.summary.scalar(
                name='learning rate',
                data=learning_rate(epoch*total_train_num//BATCH_SIZE),
                step=epoch
            )

        # save weights for every n epochs
        # if (epoch+1) % SAVE_EVERY_N_EPOCH == 0:
        #     model.save_weights(filepath=SAVED_MODEL_DIR + 'epoch-{}'.format(epoch), save_format='tf')

        # save weights when current model increases its performance on valid dataset
        if valid_loss.result().numpy() < prev_valid_loss:
            prev_valid_loss = valid_loss.result().numpy()
            model.save_weights(filepath=SAVED_MODEL_DIR + 'best', save_format='tf')

        train_loss.reset_states()
        train_acc.reset_states()
        valid_loss.reset_states()
        valid_acc.reset_states()

    # save the whole model

    tf.keras.models.save_model(model, SAVED_MODEL_DIR)
    # tf.saved_model.save(model, SAVED_MODEL_DIR)

