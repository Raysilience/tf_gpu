import tensorflow as tf
from tqdm import tqdm

from models.CNN import CNN
from models.MLP import MLP
from utils.data_loader import DataLoader
from models import ShuffleNetV2
from config import *


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices(device_type='GPU')
    tf.config.set_visible_devices(devices=gpus[0], device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
        # tf.config.set_logical_device_configuration(device=gpus[0], logical_devices=[tf.config.LogicalDeviceConfiguration(memory_limit=0)])

    # create model
    model = CNN()
    # model = ShuffleNetV2.shufflenet_0_1x()

    # create dataloader
    train_data_loader = DataLoader(TRAIN_TFRECORD)
    valid_data_loader = DataLoader(VALID_TFRECORD)

    train_dataset = train_data_loader.get_dataset(BATCH_SIZE, augment=False)
    valid_dataset = valid_data_loader.get_dataset(BATCH_SIZE)

    total_train_num = train_data_loader.get_len()


    # define loss and optimizer
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
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
        initial_learning_rate=3e-3,
        decay_steps=20,
        decay_rate=0.1,
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_obj(y_true=y, y_pred=y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
        train_loss.update_state(values=loss)
        train_acc.update_state(y_true=y, y_pred=y_pred)


    @tf.function
    def valid_step(x, y):
        y_pred = model(x, training=False)
        loss = loss_obj(y_true=y, y_pred=y_pred)
        valid_loss.update_state(values=loss)
        valid_acc.update_state(y_true=y, y_pred=y_pred)


    # 获取数据，模型预测，计算损失，计算梯度，反向传播
    for epoch in range(NUM_EPOCHS):
        step = 0
        for image_batch, label_batch in tqdm(train_dataset, desc='training'):
            train_step(image_batch, label_batch)
            # print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch,
            #                                                                          NUM_EPOCHS,
            #                                                                          step,
            #                                                                          total_train_num // BATCH_SIZE,
            #                                                                          train_loss.result().numpy(),
            #                                                                          train_acc.result().numpy()))
            step += 1
            if step > total_train_num/BATCH_SIZE + 1: break

        for image_batch, label_batch in valid_dataset:
            valid_step(image_batch, label_batch)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch,
                                                                  NUM_EPOCHS,
                                                                  train_loss.result().numpy(),
                                                                  train_acc.result().numpy(),
                                                                  valid_loss.result().numpy(),
                                                                  valid_acc.result().numpy()))


        train_loss.reset_states()
        train_acc.reset_states()
        valid_loss.reset_states()
        valid_acc.reset_states()

        # save weights for every n epochs
        if epoch % SAVE_EVERY_N_EPOCH == 0:
            model.save_weights(filepath=SAVED_MODEL_DIR + 'epoch-{}'.format(epoch), save_format='tf')

    # save weights
    # model.save_weights(filepath=SAVED_MODEL_DIR+"model", save_format='tf')

    # save model as .h5 format
    # model.save(h5_save_path)

    # save the whole model
    tf.saved_model.save(model, SAVED_MODEL_DIR)

