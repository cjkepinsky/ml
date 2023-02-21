import keras_tuner


def hyperband_search(regressor_builder, X_train, y_train, X_valid, y_valid, max_epochs=5, batch_size=128):
    # tuner = keras_tuner.RandomSearch(
    tuner = keras_tuner.Hyperband(
        hypermodel=regressor_builder,
        # The objective name and direction.
        # Name is the f"val_{snake_case_metric_class_name}".
        objective=keras_tuner.Objective("val_mean_absolute_error", direction="min"),
        max_epochs=50,
        factor=3,
        seed=42,
        # max_trials=2,
        overwrite=True,
        directory="keras_tuner"
    )

    tuner.search(
        x=X_train,
        y=y_train,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(X_valid, y_valid)
    )

    tuner.results_summary()
    # tuner.search_space_summary()
    winner = tuner.get_best_models(num_models=1)[0]
    winner.build(input_shape=[X_train.shape[1]])

    return winner
