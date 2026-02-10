from multilayer_perceptron import main

if __name__ == "__main__":
    main(
        data="newsgroups",
        num_epochs=20,
        lr=0.005,
    )
