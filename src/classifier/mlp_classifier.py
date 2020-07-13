class MlpClassifier:
    
    def __init__(self):

        LabelingDiscriminator.__init__(self, in_shape, num_classes)

        self.num_layers = num_layers
        self.layer_multiplier = layer_multiplier
        self.bn_momentum = bn_momentum
        self.dropout = dropout
        self.leaky_relu_alpha = leaky_relu_alpha

        data = Input(self.in_shape)

        flat_data = Flatten()(data)
        features = self.create_mlp_interim(
            flat_data, num_layers, leaky_relu_alpha, dropout
        )

        discrimination = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        Model.__init__(
            self,
            inputs=data,
            outputs=[discrimination, label],
            name=name or self.__class__.__name__,
        )

        self.compile(
            loss=["binary_crossentropy", "sparse_categorical_crossentropy"],
            optimizer=optimizer,
            metrics=["accuracy"],
        )
