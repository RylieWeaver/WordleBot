from train import train
from model import ActorCriticNet
from data_utils import get_vocab


def main():
    # Instantiate hyperparameters
    epochs = 50000
    lr = 1e-3
    batch_size = 100
    vocab_size = 100

    # Instantiate data
    state_size = 292  # 26 letters * 11 letter possibilities (1 for number, 5 green, 5 grey possibilites) plus 6 for one-hot of the number of guesses taken so far
    vocab = get_vocab(vocab_size=vocab_size)

    # Instantiate the network and optimizer
    actor_critic_net = ActorCriticNet(state_size=state_size, vocab_size=vocab_size, hidden_dim=128, layers=3)

    # Train the network
    train(actor_critic_net, vocab, lr, batch_size, epochs)


if __name__ == '__main__':
    main()
