import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import model_to_dot, plot_model, image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import wandb
from wandb.keras import WandbCallback
import seaborn as sns


class ModelWrapper:
    def __init__(self, data_dir, img_size, batch_size, use_generator=True, sample_datasets=False):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.use_generator = use_generator

        if use_generator:
            self._load_generated_data()
        else:
            self._load_data(sample_datasets)

    def _load_data(self, sample_datasets):
        if (sample_datasets):
            self.train_data = image_dataset_from_directory(
                self.data_dir + "/train",
                image_size=(self.img_size,   self.img_size),
                validation_split=0.2,
                subset="validation",
                seed=123,
                shuffle=True,
                batch_size=self.batch_size)

            self.val_data = image_dataset_from_directory(
                self.data_dir + "/validation",
                image_size=(self.img_size,   self.img_size),
                validation_split=0.2,
                subset="validation",
                seed=123,
                shuffle=True,
                batch_size=self.batch_size)

        else:
            self.train_data = image_dataset_from_directory(
                self.data_dir + "/train",
                image_size=(self.img_size,   self.img_size),
                seed=123,
                shuffle=True,
                batch_size=self.batch_size)

            self.val_data = image_dataset_from_directory(
                self.data_dir + "/validation",
                image_size=(self.img_size,   self.img_size),
                seed=123,
                shuffle=True,
                batch_size=self.batch_size)

        self.test_data = image_dataset_from_directory(
            self.data_dir + "/test",
            image_size=(self.img_size,   self.img_size),
            batch_size=self.batch_size)

    def _load_generated_data(self):
        """
            Loads traning, validation and test data from the data directory.
        """

        # Loads training data
        train_datagen = ImageDataGenerator(
            rescale=1/255,  # rescale the tensor values to [0,1]
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        self.train_generator = train_datagen.flow_from_directory(
            directory=f'{self.data_dir}/train',
            target_size=(self.img_size, self.img_size),
            color_mode="rgb",
            class_mode="binary",
            batch_size=self.batch_size,
            shuffle=True,
        )

        # loads validation data
        val_test_datagen = ImageDataGenerator(
            rescale=1/255  # rescale the tensor values to [0,1]
        )
        self.val_generator = val_test_datagen.flow_from_directory(
            directory=f'{self.data_dir}/validation',
            target_size=(self.img_size, self.img_size),
            color_mode="rgb",
            class_mode="binary",
            batch_size=self.batch_size,
            shuffle=True
        )

        # loads test data
        self.test_generator = val_test_datagen.flow_from_directory(
            directory=f"{self.data_dir}/test",
            classes=['REAL', 'FAKE'],
            target_size=(self.img_size, self.img_size),
            color_mode="rgb",
            class_mode=None,
            batch_size=1,
            shuffle=False
        )

    def create_model(self, model_file, layers, config, use_wandb=True):
        """
            Creates a model .
        """
        self.model = Sequential(layers=layers)
        self.model_file = model_file
        self.custom_callbacks = [
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=5,
                verbose=1
            ),
        ]
        if use_wandb:
            self.custom_callbacks.append(WandbCallback(data_type="image"))
        if model_file != None:
            self.custom_callbacks.append(
                ModelCheckpoint(
                    filepath=model_file,
                    monitor='val_loss',
                    mode='min',
                    verbose=1,
                    save_best_only=True
                ),
            )

        self.config = config

    def fit(self):
        """
            Fits the model and returns history
        """
        train_data = self.train_generator if self.use_generator else self.train_data
        val_data = self.val_generator if self.use_generator else self.val_data
        return self.model.fit(
            train_data,
            epochs=self.config.epochs,
            steps_per_epoch=len(train_data),
            validation_data=val_data,
            validation_steps=len(val_data),
            callbacks=self.custom_callbacks,
            use_multiprocessing=True,
        )

    def evaluate_model(self, best_model):
        test_data = self.test_generator if self.use_generator else self.test_data
        preds = best_model.predict(
            test_data,
            verbose=1
        )
        if self.use_generator:
            print("self.test_generator.filenames: ",
                  len(self.test_generator.filenames))
            print("preds: ", len(preds.flatten()))
            test_results = pd.DataFrame({
                "Filename": self.test_generator.filenames,
                "Prediction": preds.flatten()
            })

        test_results["Rounded"] = test_results["Prediction"].round()

        # creates confusion matrix
        true_positive_fake = test_results[(test_results['Filename'].str.startswith(
            'FAKE')) & (test_results['Rounded'] == 0)].count()[0]
        false_positive_fake = test_results[(test_results['Filename'].str.startswith(
            'REAL')) & (test_results['Rounded'] == 0)].count()[0]

        true_positive_real = test_results[(test_results['Filename'].str.startswith(
            'REAL')) & (test_results['Rounded'] == 1)].count()[0]
        false_positive_real = test_results[(test_results['Filename'].str.startswith(
            'FAKE')) & (test_results['Rounded'] == 1)].count()[0]

        conf_matrix = np.matrix([
            [true_positive_fake, false_positive_fake],
            [false_positive_real, true_positive_real]
        ])
        print("conf_matrix: \n", conf_matrix)
        sns.heatmap(conf_matrix, square=True, annot=True,
                    cmap='Reds', fmt='d', cbar=False)

        # Log the resultmatrix to wandb
        wandb.log({
            'true_positive_fake': true_positive_fake,
            'false_positive_fake': false_positive_fake,
            'true_positive_real': true_positive_real,
            'false_positive_real': false_positive_real
        })

    def export_to_png(self):
        # Save model to png
        plot_model(
            self.model, show_shapes=True, show_layer_names=True, to_file=f'images/{self.model_file}_model.png')
