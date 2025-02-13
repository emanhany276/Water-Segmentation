import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.data import AUTOTUNE
from tensorflow.keras.applications.resnet50 import preprocess_input

# Set parameters
img_size = (224, 224)
batch_size = 32

# Load dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"D:\Task_1\Teeth Classification\Teeth_Dataset\Training",
    image_size=img_size,
    batch_size=batch_size,
    color_mode="rgb"
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"D:\Task_1\Teeth Classification\Teeth_Dataset\Testing",
    image_size=img_size,
    batch_size=batch_size,
    color_mode="rgb"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"D:\Task_1\Teeth Classification\Teeth_Dataset\Validation",
    image_size=img_size,
    batch_size=batch_size,
    color_mode="rgb"
)

class_names = train_ds.class_names

# ✅ Data Augmentation
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),  # Reduce rotation
    keras.layers.RandomZoom(0.15),  # Reduce zoom
    keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),  # Reduce shift
    keras.layers.RandomBrightness(0.15),
    keras.layers.RandomContrast(0.15),
])

# ✅ Preprocessing Function
def preprocess(image, label, augment=False):
    if augment:
        image = data_augmentation(image, training=True)

    image = preprocess_input(image)  # ResNet50 expects [-1,1] normalization
    return image, label

# ✅ Train Data Preparation
def train_data_prep(data, shuffle_size):
    data = data.map(lambda x, y: preprocess(x, y, augment=True))
    data = data.shuffle(shuffle_size).cache().repeat().prefetch(AUTOTUNE)
    return data

# ✅ Test/Validation Data Preparation
def test_data_prep(data):
    data = data.map(preprocess).cache().prefetch(AUTOTUNE)
    return data

# Apply Data Preprocessing
train_data_prepared = train_data_prep(train_ds, shuffle_size=1000)
test_data_prepared = test_data_prep(test_ds)
val_data_prepared = test_data_prep(val_ds)

# ✅ Load ResNet50 Model (Without Top Layers)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze layers for initial training

# ✅ Build the Classification Model
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Regularization
outputs = Dense(len(class_names), activation="softmax")  # Multi-class classification

outputs = Dense(len(class_names), activation="softmax")(x)  # ✅ Connect to x


# ✅ Define Model
model = Model(inputs=inputs, outputs=outputs)  

# ✅ Debugging Step: Check Model Summary
print(model.summary())  # Ensure model is created before compiling

# ✅ Compile Model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ✅ Train Initial Model
history = model.fit(
    train_data_prepared,
    validation_data=val_data_prepared,
    steps_per_epoch=len(train_ds),
    validation_steps=len(val_ds),
    epochs=10
)

# ✅ Fine-Tune Model: Unfreeze Last Layers
base_model.trainable = True
for layer in base_model.layers[:100]:  # Keep first 100 layers frozen
    layer.trainable = False

# ✅ Recompile with Lower Learning Rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ✅ Fine-Tune Model
history_fine = model.fit(
    train_data_prepared,
    validation_data=val_data_prepared,
    steps_per_epoch=len(train_ds),
    validation_steps=len(val_ds),
    epochs=10
)

model.save("fine_tuned.h5")

# ✅ Evaluate on Test Data
test_loss, test_acc = model.evaluate(test_data_prepared)
print(f"Test Accuracy: {test_acc:.4f}")

