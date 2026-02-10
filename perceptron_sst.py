from perceptron import main

if __name__ == "__main__":
    main(
        data="sst2",
        features="bow+len",
        num_epochs=7,
        lr=0.001,
    )
