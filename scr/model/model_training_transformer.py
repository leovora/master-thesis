from keras.models import Model
from keras.layers import (
    Add,
    Dense,
    Dropout,
    Embedding,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Lambda,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import os
from scr.data.load_data import preprocess_transformer_data, load_data_from_csv
import numpy as np


def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):

    attention_output = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout,
    )(
        inputs,
        inputs,
        use_causal_mask=True,
    )

    attention_output = Dropout(dropout)(attention_output)

    x = Add()([inputs, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff_output = Dense(ff_dim, activation="gelu")(x)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)

    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    return x


def build_model(sequence_length, feature_dim):

    model_input = Input(shape=(sequence_length, feature_dim))

    # IMPORTANTISSIMO:
    # proiezione iniziale in spazio latente più grande
    x = Dense(64)(model_input)

    # positional embedding
    positions = tf.range(start=0, limit=sequence_length, delta=1)

    positional_embedding = Embedding(
        input_dim=sequence_length,
        output_dim=64,
    )(positions)

    positional_embedding = Lambda(
        lambda t: tf.expand_dims(t, axis=0)
    )(positional_embedding)

    x = Add()([x, positional_embedding])

    # transformer blocks
    x = transformer_block(
        x,
        head_size=64,
        num_heads=4,
        ff_dim=256,
        dropout=0.1,
    )

    x = transformer_block(
        x,
        head_size=64,
        num_heads=4,
        ff_dim=256,
        dropout=0.1,
    )

    # MOLTO meglio del GlobalAveragePooling
    # equivalente al return_sequences=False dell'LSTM
    x = Lambda(lambda t: t[:, -1, :])(x)

    x = Dense(128, activation="gelu")(x)
    x = Dropout(0.1)(x)

    x = Dense(64, activation="gelu")(x)

    output = Dense(1)(x)

    model = Model(inputs=model_input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="mean_squared_error",
    )

    return model


def train_model(X, y, sequence_length):

    val_size = max(1, int(len(X) * 0.2))

    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    model = build_model(sequence_length, X.shape[2])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=16,
        epochs=50,
        shuffle=False,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    return model


def train_and_save_models(
    tickers,
    sequence_length,
    model_folder: str = "models",
    data_folder: str = "../data",
    force_retrain=False,
):
    models = {}
    for ticker in tickers:
        model_path = os.path.join(model_folder, f"{ticker}_model.h5")

        if os.path.exists(model_path) and not force_retrain:
            print(f"Model for {ticker} already exists. Skipping training.")
            continue

        print(f"Processing {ticker}...")

        try:
            df = load_data_from_csv(ticker, data_folder=data_folder)
            if df.empty:
                print(f"No data found for {ticker}. Skipping...")
                continue

            (X_train, y_train, X_test, y_test, _, _, _, _) = preprocess_transformer_data(df, sequence_length)

            X = np.concatenate([X_train, X_test], axis=0)
            y = np.concatenate([y_train, y_test], axis=0)

            if len(X) == 0:
                print(f"No data found for {ticker}. Skipping...")
                continue

            print(
                f"Training {ticker}: samples={len(X)}, "
                f"sequence_length={sequence_length}, features={X.shape[2]}"
            )
            model = train_model(X, y, sequence_length)
            models[ticker] = model

            model.save(model_path)
            print(f"Model for {ticker} saved.")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    return models
