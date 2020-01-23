from model import Modified_U_Net, SNR
from utils import generate_data
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# generate training data of ellipses
Y_train, X_train = generate_training_data(image_size=128, number_of_images=3200, number_of_angles=50)

# print few samples from the training set
print_data_samples(Y_train, X_train, n_samples=4)

# create an instance of the modified U-net model
model = Modified_U_Net(input_size=128, input_channels=1, filters=16, learning_rate=0.001, scale=0.01)

# set up checkpoint and earlystopping
checkpoint = ModelCheckpoint(filepath='model.ckpt', save_best_only=True, monitor='val_SNR', mode='max', verbose=1)
earlystopping = EarlyStopping(monitor='val_SNR', mode='max', verbose=1, patience=50, restore_best_weights=True)

# train the model
model.fit(x=X_train, y=Y_train, validation_split=0.05, batch_size=32, epochs=100, callbacks=[checkpoint, earlystopping])

# print few results of outputs of the model on the training set
print_model_outputs(model, Y_train, X_train, n_samples=4)

# generate training data of rectangles
Y_test, X_test = generate_testing_data(image_size=128, number_of_images=200, number_of_angles=50)

# compute average SNR on the test set
print('Average SNR on the test set: {}.'.format(SNR(Y_test, model(X_test))))

# print few results of outputs of the model on the test set
print_model_outputs(model, Y_test, X_test, n_samples=4)
