import math
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

from utils import model_loader
from scripts.config import *
from utils.data_loader import DataLoader

if __name__ == '__main__':
    # GPU settings
    # ==========================================================================

    gpus = tf.config.list_physical_devices(device_type='GPU')
    tf.config.set_visible_devices(devices=gpus[0:2], device_type='GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(device=gpus[0:2], enable=True)
    # tf.config.set_logical_device_configuration(
    #     device=gpus[0],
    #     logical_devices=[tf.config.LogicalDeviceConfiguration(memory_limit=0)]
    # )

    # create model
    # ==========================================================================
    model = model_loader.load(
        mode=1,
        model_name='MobileNetV2',
        filepath=SAVED_MODEL_DIR+'best',
        dirpath=SAVED_MODEL_DIR
    )

    # create dataloader
    # ==========================================================================
    train_data_loader = DataLoader(TRAIN_TFRECORD)
    valid_data_loader = DataLoader(VALID_TFRECORD)

    train_dataset = train_data_loader.get_dataset(BATCH_SIZE, augment=True)
    valid_dataset = valid_data_loader.get_dataset(BATCH_SIZE)

    total_train_num = train_data_loader.get_len()

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
        initial_learning_rate=1e-3,
        decay_steps=50,
        decay_rate=0.1,
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss0_obj = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
    prev_valid_loss = float('inf')

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss0_obj(y_true=y, y_pred=y_pred)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
        train_loss.update_state(values=loss)
        train_acc.update_state(y_true=y, y_pred=y_pred)


    @tf.function
    def valid_step(x, y):
        y_pred = model(x, training=False)
        loss = loss0_obj(y_true=y, y_pred=y_pred)

        valid_loss.update_state(values=loss)
        valid_acc.update_state(y_true=y, y_pred=y_pred)


    # training process
    # ==========================================================================
    for epoch in range(NUM_EPOCHS):
        step = 0
        for image_batch, label_batch in tqdm(train_dataset, desc='training'):
            train_step(image_batch, label_batch)
            step += 1
            if step > math.ceil(total_train_num/BATCH_SIZE):
                break

        for image_batch,  label_batch in valid_dataset:
            valid_step(image_batch, label_batch)

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

