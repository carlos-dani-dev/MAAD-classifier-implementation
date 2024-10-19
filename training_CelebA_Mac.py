import CelebA_Mac as cm


celebA_embeddings_path = ""
celebA_embeddings_filenames_path = ""

celebA_embeddings, celebA_labels, celebA_filenames = cm.load_embeddings(celebA_embeddings_path,
                                                                        celebA_embeddings_filenames_path)
train, test, valid = cm.manipulating_embeddings(celebA_embeddings, celebA_labels, celebA_filenames)
trainX, trainy_model_input, train_filenames = train[0], train[1], train[2]
testX, testy_model_input, test_filenames = test[0], test[1], test[2]
validX, validy_model_input, valid_filenames = valid[0], valid[1], valid[2]
model = cm.multitask_model_definition(attributes_qtt=40)
cm.compiling_model(model, attributes_qtt=40)
history = cm.training_model(model, trainX, testX, trainy_model_input, testy_model_input, batch_size=128, epochs=100)

print(history)