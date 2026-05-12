"""Project-wide constants — all names, keys, and colour codes live here."""

MODEL_FCN = "fcn"
MODEL_RNN = "rnn"
MODEL_LSTM = "lstm"
MODEL_NAMES = [MODEL_FCN, MODEL_RNN, MODEL_LSTM]

INPUT_SIZE_FCN = 15      # window(10) + selector(5)
SEQ_INPUT_SIZE = 6       # window_feature(1) + selector(5) broadcast per timestep
WINDOW_SIZE = 10
SELECTOR_SIZE = 5

DATASET_KEY_X_TRAIN = "X_train"
DATASET_KEY_X_VAL = "X_val"
DATASET_KEY_X_TEST = "X_test"
DATASET_KEY_C_TRAIN = "C_train"
DATASET_KEY_C_VAL = "C_val"
DATASET_KEY_C_TEST = "C_test"
DATASET_KEY_Y_TRAIN = "y_train"
DATASET_KEY_Y_VAL = "y_val"
DATASET_KEY_Y_TEST = "y_test"

RESULT_COL_MODEL = "model"
RESULT_COL_TRAIN_MSE = "train_mse"
RESULT_COL_VAL_MSE = "val_mse"
RESULT_COL_TEST_MSE = "test_mse"
RESULT_COL_EPOCHS = "epochs_trained"
RESULT_COL_STOPPED_EARLY = "stopped_early"

PLOT_COLOR_CLEAN = "green"
PLOT_COLOR_NOISY = "grey"
PLOT_COLOR_FCN = "red"
PLOT_COLOR_RNN = "orange"
PLOT_COLOR_LSTM = "purple"
