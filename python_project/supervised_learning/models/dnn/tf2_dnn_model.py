train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  for x, y in train_dataset:
    with tf.GradientTape() as tape:
        loss_value = tf.keras.losses.SparseCategoricalCrossentropy(y, a, training=True)

    loss_value, tape.gradient(loss_value, model.trainable_variables)
    tf.keras.optimizers.SGD(learning_rate=0.01).apply_gradients(zip(grads, model.trainable_variables))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
