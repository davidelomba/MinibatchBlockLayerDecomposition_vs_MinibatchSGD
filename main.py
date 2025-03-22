import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
from itertools import product
import time
import random

# Parametri globali
BATCH_SIZE = 64
INPUT_SIZE = 28 * 28
HIDDEN_SIZES = [128, 64, 32]
OUTPUT_SIZE = 10
EPOCHS = 150  # Epoche
LR = 0.5  # Learning Rate
EPS = 0.001  # Parametro che riduce il LR ad ogni batch
RHO = 0.00001  # Parametro per la regolarizzazione della loss
GROUP_SIZE = 2



# Rete neurale fully connected
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=10):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], output_size)
        )

    def forward(self, x):
        return self.model(x)

# Caricamento dataset MNIST e trasformazione in tensori
def load_dataset():
    ds_train = MNIST(root='./data', download=True, train=True)
    ds_test = MNIST(root='./data', download=True, train=False)

    Xs_train = ds_train.data.to(torch.float32)
    ys_train = ds_train.targets
    Xs_test = ds_test.data.to(torch.float32)
    ys_test = ds_test.targets

    return Xs_train, ys_train, Xs_test, ys_test

# Normalizzazione dei campioni di train e di test
def normalize_dataset(Xs_train, Xs_test, ys_train, ys_test, device):
    mean_px = Xs_train.mean()   # Calcolo della media
    std_px = Xs_train.std()     # Calcolo della deviazione standard
    Xs_train = (Xs_train - mean_px) / std_px    # Normalizzazione secondo Z-score
    Xs_test = (Xs_test - mean_px) / std_px

    Xs_train = Xs_train.flatten(1, 2)       # Trasformazione del tensore (num_campioni, 28, 28) in (num_campioni, 784)
    Xs_test = Xs_test.flatten(1, 2)

    Xs_test = Xs_test.to(device)
    ys_test = ys_test.to(device)

    Xs_train = Xs_train.to(device)
    ys_train = ys_train.to(device)

    return Xs_train, ys_train, Xs_test, ys_test

# Estrazione di un subset dal set di train
def subset_train(Xs_train, train_size):
    I = np.random.permutation(range(len(Xs_train)))[:train_size]
    Xs_train_ss = Xs_train[I]
    ys_train_ss = ys_train[I]
    return Xs_train_ss, ys_train_ss

# Creazione dei batch tramite DataLoader
def create_batch(batch_size, Xs_train_ss, ys_train_ss, Xs_val, ys_val, Xs_test, ys_test):
    ds_train = TensorDataset(Xs_train_ss, ys_train_ss)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    ds_val = TensorDataset(Xs_val, ys_val)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    ds_test = TensorDataset(Xs_test, ys_test)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    return dl_train, dl_val, dl_test


# Addestramento della rete secondo l'algoritmo Minibatch Block Layer Decomposition (MBLD)
def train_model_decomposition(model, dl_train, dl_val, epochs, lr, eps, rho, batch_size):
    a = lr
    avg_losses = []     # Lista che mantiene la media della loss per ogni epoca
    losses = []         # Lista che contiene la loss dell'ultimo batch
    times = []          # Lista che contiene i tempi impiegati per ogni epoca
    val_losses = []     # Lista delle loss di validazione
    val_accuracies = [] # Lista delle accuracy di validazione
    total_time = 0
    criterion_loss = nn.CrossEntropyLoss()      # Utilizzo la CrossEntropy come loss
    layers = [layer for layer in model.model if isinstance(layer, nn.Linear)]        # Considero solo i layer nn.Linear
    optimizers = [optim.SGD([layer.weight, layer.bias], lr=a)       # Creazione di un ottimizzatore per layer
                  for layer in model.model if isinstance(layer, nn.Linear)]


    model.train()
    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        for batch in dl_train:
            x, y = batch
            model.zero_grad()       # Azzero i gradienti
            for layer, optimizer in zip(reversed(layers), reversed(optimizers)):       # Itero i layer dall'ultimo al primo
                optimizer.zero_grad()
                y_pred = model(x)        # Calcolo il valore predetto
                loss = criterion_loss(y_pred, y)     # Calcolo la loss
                l2_penalty = sum(torch.sum(param ** 2) for param in layer.parameters())     # Applico un regolarizzatore considerando solo i pesi del layer
                loss = loss + rho * batch_size * l2_penalty
                total_loss += loss.item()
                loss.backward()        # Calcolo del gradiente
                optimizer.step()        # Aggiornamento dei pesi

            a = a * (1 - eps * a)       # Aggiornamento del learning rate
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = a
        end_time = time.time()
        epoch_time = end_time - start_time      # Calcolo il tempo impiegato per un epoca
        total_time += epoch_time
        times.append(epoch_time)
        avg_loss = total_loss / len(dl_train)
        losses.append(loss.item())
        avg_losses.append(avg_loss)

        # Validazione del modello
        val_loss, val_acc = evaluate_model(model, dl_val, mode="Validation")
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"\nEpoch {epoch + 1}: Train Loss: {avg_loss:.6f}, Time: {epoch_time:.4f} s")

    print(f"Total time: {total_time:.4f} s")
    return avg_losses, losses, times, val_losses, val_accuracies

def train_model_decomposition_ada(model, dl_train, dl_val, epochs, lr, rho, batch_size):
    avg_losses = []     # Lista che mantiene la media della loss per ogni epoca
    losses = []         # Lista che contiene la loss dell'ultimo batch
    times = []          # Lista che contiene i tempi impiegati per ogni epoca
    val_losses = []     # Lista delle loss di validazione
    val_accuracies = [] # Lista delle accuracy di validazione
    total_time = 0
    criterion_loss = nn.CrossEntropyLoss()      # Utilizzo la CrossEntropy come loss
    layers = [layer for layer in model.model if isinstance(layer, nn.Linear)]        # Considero solo i layer nn.Linear
    optimizers = [optim.Adagrad([layer.weight, layer.bias], lr=lr)       # Creazione di un ottimizzatore per layer
                  for layer in model.model if isinstance(layer, nn.Linear)]


    model.train()
    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        for batch in dl_train:
            x, y = batch
            model.zero_grad()       # Azzero i gradienti
            for layer, optimizer in zip(reversed(layers), reversed(optimizers)):       # Itero i layer dall'ultimo al primo
                optimizer.zero_grad()
                y_pred = model(x)        # Calcolo il valore predetto
                loss = criterion_loss(y_pred, y)     # Calcolo la loss
                l2_penalty = sum(torch.sum(param ** 2) for param in layer.parameters())     # Applico un regolarizzatore considerando solo i pesi del layer
                loss = loss + rho * batch_size * l2_penalty
                total_loss += loss.item()
                loss.backward()        # Calcolo del gradiente
                optimizer.step()        # Aggiornamento dei pesi

        end_time = time.time()
        epoch_time = end_time - start_time      # Calcolo il tempo impiegato per un epoca
        total_time += epoch_time
        times.append(epoch_time)
        avg_loss = total_loss / len(dl_train)
        losses.append(loss.item())
        avg_losses.append(avg_loss)

        # Validazione del modello
        val_loss, val_acc = evaluate_model(model, dl_val, mode="Validation")
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"\nEpoch {epoch + 1}: Train Loss: {avg_loss:.6f}, Time: {epoch_time:.4f} s")

    print(f"Total time: {total_time:.4f} s")
    return avg_losses, losses, times, val_losses, val_accuracies

# Addestramento della rete secondo l'algoritmo Minibatch Block Layer Decomposition (MBLD) considerando un sottoinsieme dei layers
def train_model_decomposition_sub(model, dl_train, dl_val, epochs, lr, eps, rho, group_size, batch_size):
    a = lr
    avg_losses = []  # Lista che mantiene la media della loss per ogni epoca
    losses = []  # Lista che contiene la loss dell'ultimo batch
    times = []  # Lista che contiene i tempi impiegati per ogni epoca
    val_losses = []     # Lista delle loss di validazione
    val_accuracies = [] # Lista delle accuracy di validazione
    total_time = 0
    criterion_loss = nn.CrossEntropyLoss()  # Utilizzo la CrossEntropy come loss

    layers = [layer for layer in model.model if isinstance(layer, nn.Linear)]

    # Creazione di ottimizzatori per i gruppi di layer
    optimizers = []
    for i in range(len(layers) - group_size + 1, -1, -group_size):
        group_layers = layers[i:i + group_size]            # Selezione del gruppo di layer
        params = []
        for layer in group_layers:              # Creazione di un ottimizzatore per il gruppo di layer
            params.extend([layer.weight, layer.bias])       # Aggiunta dei pesi e i bias del gruppo
        optimizer = optim.SGD(params, lr=a)
        optimizers.append(optimizer)

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()

        for batch in dl_train:
            x, y = batch
            model.zero_grad()

            for idx, optimizer in enumerate(optimizers):
                y_pred = model(x)       # Calcolo del valore predetto
                loss = criterion_loss(y_pred, y)        # Calcolo della loss
                l2_penalty = sum(torch.sum(param ** 2) for param in
                                 optimizer.param_groups[0]['params'])       # Applica un regolarizzatore
                loss = loss + rho * batch_size * l2_penalty
                total_loss += loss.item()

                # Calcola il gradiente e aggiorna i pesi
                loss.backward()
                optimizer.step()

                # Aggiornamento del learning rate
            a = a * (1 - eps * a)
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = a

        # Calcolo del tempo per un'epoca e media della loss
        end_time = time.time()
        epoch_time = end_time - start_time
        total_time += epoch_time
        times.append(epoch_time)
        avg_loss = total_loss / len(dl_train)
        losses.append(loss.item())
        avg_losses.append(avg_loss)

        # Validazione del modello
        val_loss, val_acc = evaluate_model(model, dl_val, mode="Validation")
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"\nEpoch {epoch + 1}: Train Loss: {avg_loss:.6f}, Time: {epoch_time:.4f} s")

    print(f"Total time: {total_time:.4f} s")
    return avg_losses, losses, times, val_losses, val_accuracies


# Addestramento della rete secondo l'algoritmo Minibatch Block Layer Decomposition (MBLD) con scelta casuale del batch
def train_model_decomposition_rnd(model, dl_train, dl_val, epochs, lr, eps, rho, batch_size):
    a = lr
    avg_losses = []     # Lista che mantiene la media della loss per ogni epoca
    losses = []         # Lista che contiene la loss dell'ultimo batch
    times = []          # Lista che contiene i tempi impiegati per ogni epoca
    val_losses = []     # Lista delle loss di validazione
    val_accuracies = [] # Lista delle accuracy di validazione
    total_time = 0
    criterion_loss = nn.CrossEntropyLoss()      # Utilizzo la CrossEntropy come loss
    layers = [layer for layer in model.model if isinstance(layer, nn.Linear)]        # Considero solo i layer nn.Linear
    optimizers = [optim.SGD([layer.weight, layer.bias], lr=a)       # Creazione di un ottimizzatore per layer
                  for layer in model.model if isinstance(layer, nn.Linear)]

    model.train()
    batch_list = list(dl_train)

    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        random.shuffle(batch_list)
        for batch in batch_list:
            x, y = batch
            model.zero_grad()       # Azzero i gradienti
            for layer, optimizer in zip(reversed(layers), reversed(optimizers)):       # Itero i layer dall'ultimo al primo
                optimizer.zero_grad()
                y_pred = model(x)        # Calcolo il valore predetto
                loss = criterion_loss(y_pred, y)     # Calcolo la loss
                l2_penalty = sum(torch.sum(param ** 2) for param in layer.parameters())     # Applico un regolarizzatore considerando solo i pesi del layer
                loss = loss + rho * batch_size * l2_penalty
                total_loss += loss.item()
                loss.backward()        # Calcolo del gradiente
                optimizer.step()        # Aggiornamento dei pesi

            a = a * (1 - eps * a)       # Aggiornamento del learning rate
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = a
        end_time = time.time()
        epoch_time = end_time - start_time      # Calcolo il tempo impiegato per un epoca
        total_time += epoch_time
        times.append(epoch_time)
        avg_loss = total_loss / len(dl_train)
        losses.append(loss.item())
        avg_losses.append(avg_loss)

        # Validazione del modello
        val_loss, val_acc = evaluate_model(model, dl_val, mode="Validation")
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"\nEpoch {epoch + 1}: Train Loss: {avg_loss:.6f}, Time: {epoch_time:.4f} s")

    print(f"Total time: {total_time:.4f} s")
    return avg_losses, losses, times, val_losses, val_accuracies

# Addestramento della rete secondo l'algoritmo Minibatch SGD
def train_model_minibatch(model, dl_train, dl_val, epochs, lr, eps, rho, batch_size):
    a = lr
    losses = []
    avg_losses = []
    times = []
    val_losses = []
    val_accuracies = []
    total_time = 0
    criterion_loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=a)

    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        for batch in dl_train:
            x, y = batch
            optimizer.zero_grad()

            y_pred = model(x)

            loss = criterion_loss(y_pred, y)
            l2_penalty = 0
            for param in model.parameters():
                l2_penalty += torch.sum(param ** 2)
            loss = loss + rho * batch_size * l2_penalty
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            a = a * (1 - eps * a)
            for param_group in optimizer.param_groups:
                param_group['lr'] = a

        end_time = time.time()
        epoch_time = end_time - start_time
        total_time += epoch_time
        times.append(epoch_time)
        avg_loss = total_loss / len(dl_train)
        losses.append(loss.item())
        avg_losses.append(avg_loss)

        # Validazione del modello
        val_loss, val_acc = evaluate_model(model, dl_val, mode="Validation")
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"\nEpoch {epoch + 1}: Train Loss: {avg_loss:.6f}, Time: {epoch_time:.4f} s")

    print(f"Total time: {total_time:.4f} s")
    return avg_losses, losses, times, val_losses, val_accuracies


# Funzione per che applica il test e la validazione del modello
def evaluate_model(model, dataloader, mode = "Validation"):
    model.eval()
    correct = 0     # Mantiene il conto dei campioni correttamente predetti
    total = 0       # Conta tutti i campioni testati
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0     # Somma la loss di ogni batch

    with torch.no_grad():
        for x, y in dataloader:

            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()

            _, predicted = torch.max(y_pred, dim=1)      # Considera la classe con probabilità più alta
            for i in range(len(predicted)):     # Controlla se le etichette predette sono corrette
                if predicted[i] == y[i]:
                    correct += 1
            total += y.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)

    print(f"{mode} Loss: {avg_loss:.6f}, {mode} Accuracy: {accuracy * 100:.2f}%")
    return avg_loss, accuracy

# Crea il grafico della media delle loss in relazione alle epoche per i due modelli
def plot_avg_loss(avg1, avg2, label1, label2):
    plt.plot(avg1, label=label1)
    plt.plot(avg2, label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Crea il grafico delle loss in relazione alle epoche per i due modelli
def plot_loss(losses1, losses2, label1, label2):
    plt.plot(losses1, label=label1)
    plt.plot(losses2, label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Crea il grafico del tempo impiegato in relazione alle epoche per i due modelli
def plot_time(times1, times2, label1, label2):
    plt.plot(times1, label=label1)
    plt.plot(times2, label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('Time')
    plt.legend()
    plt.grid(True)
    plt.show()


# funzione per la ricerca dei parametri migliori (quelli che minimizzano la loss del validation set)
def grid_search(dl_train, dl_val, model_grid, alpha=1, device='cpu'):
    # Parametri che verrano testati in combinazione
    param_grid = {
        'lr': [0.5, 0.1, 0.05, 0.01],
        'eps': [0.001, 0.0001],
        'rho': [0.0, 0.001, 0.0001, 0.00001],
    }

    best_score = float('inf')
    best_params = None

    for lr, eps, rho in product(*param_grid.values()):
        print(f"\n \nTesting combination: lr={lr}, eps={eps}, rho={rho}")

        # Creazione modello
        model = model_grid(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE).to(device)

        # Training del modello con i parametri selezionati
        avg_losses, _, _, val_losses, val_accuracies = train_model_minibatch(
            model, dl_train, dl_val, EPOCHS, lr, eps, rho, BATCH_SIZE
        )

        # Considero la loss e l'accuracy dell'ultima epoca
        final_loss = avg_losses[-1]
        val_loss = val_losses[-1]
        val_accuracy = val_accuracies[-1]

        print(f"Training Final Loss: {final_loss}")
        print(f"Validation Loss: {val_loss}")
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        # Score basato solo sulla validazione
        score = alpha * val_loss + (1 - alpha) * (1 - val_accuracy)

        print(f"Score: {score}")

        if score < best_score:
            best_score = score
            best_params = {
                'lr': lr,
                'eps': eps,
                'rho': rho,
            }

    print(f"\nBest parameters: {best_params}")
    print(f"Best Score: {best_score}")
    return best_params, best_score

# Metodo che confronta MBDL con Minibatch SGD
def mbld_vs_minibatch(dl_train, dl_val, device):
    # Creazione e addestramento di un modello MBLD
    model_decomposition = MLP(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
    model_decomposition.to(device)
    avg_losses_decomposition, losses_decomposition, times_decomposition, val_losses_decomposition, val_accuracy_decomposition = train_model_decomposition(model_decomposition,
                                                                                                    dl_train, dl_val, EPOCHS,
                                                                                                    LR, EPS, RHO,
                                                                                                    GROUP_SIZE)

    # Creazione e addestramento di un modello Minibatch SGD
    model_minibatch = MLP(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
    model_minibatch.to(device)
    avg_losses_minibatch, losses_minibatch, times_minibatch, val_losses_minibatch, val_accuracy_minibatch = train_model_minibatch(model_minibatch, dl_train, dl_val, EPOCHS,
                                                                                    LR, EPS, RHO, BATCH_SIZE)

    # Visualizzazione dei grafici
    plot_avg_loss(avg_losses_decomposition, avg_losses_minibatch, 'MBLD', 'Minibatch SGD')
    plot_loss(losses_decomposition, losses_minibatch, 'MBLD', 'Minibatch SGD')

    # Stampa dell'accuratezza e della loss dei due modelli
    print("Testing MBLD ...")
    test_loss_decomposition, test_accuracy_decomposition = evaluate_model(model_decomposition, dl_test, mode="Test")
    print("\nTesting Minibatch GD ...")
    test_loss_minibatch, test_accuracy_minibatch = evaluate_model(model_minibatch, dl_test, mode="Test")

    return test_loss_decomposition, test_accuracy_decomposition, test_loss_minibatch, test_accuracy_minibatch


# Metodo che confronta MBDL e MBLD con Adagrad

def mbld_vs_ada(dl_train, dl_val, device):
    # Creazione e addestramento di un modello MBLD
    model_decomposition = MLP(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
    model_decomposition.to(device)
    avg_losses_decomposition, losses_decomposition, times_decomposition, val_losses_decomposition, val_accuracy_decomposition = train_model_decomposition(model_decomposition,
                                                                                                    dl_train, dl_val, EPOCHS,
                                                                                                    LR, EPS, RHO,
                                                                                                    GROUP_SIZE)

    # Creazione e addestramento di un modello MBLD con Adagrad
    model_ada = MLP(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
    model_ada.to(device)
    avg_losses_ada, losses_ada, times_ada, val_losses_ada, val_accuracy_ada = train_model_decomposition_ada(model_ada, dl_train, dl_val, EPOCHS,
                                                                                    LR, RHO, BATCH_SIZE)

    # Visualizzazione dei grafici
    plot_avg_loss(avg_losses_decomposition, avg_losses_ada, 'MBLD', 'MBLD with Adagrad')
    plot_loss(losses_decomposition, losses_ada, 'MBLD', 'MBLD with Adagrad')

    # Stampa dell'accuratezza e della loss dei due modelli
    print("Testing MBLD ...")
    test_loss_decomposition, test_accuracy_decomposition = evaluate_model(model_decomposition, dl_test, mode = "Test")
    print("\nTesting MBLD with Adagrad ...")
    test_loss_ada, test_accuracy_ada = evaluate_model(model_ada, dl_test, mode = "Test")

    return test_loss_decomposition, test_accuracy_decomposition, test_loss_ada, test_accuracy_ada


# Metodo che confronta MBDL e MBLD con la sequenza di batch che cambia ad ogni epoca

def mbld_vs_rnd(dl_train, dl_val, device):
    # Creazione e addestramento di un modello MBLD
    model_decomposition = MLP(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
    model_decomposition.to(device)
    avg_losses_decomposition, losses_decomposition, times_decomposition, val_losses_decomposition, val_accuracy_decomposition = train_model_decomposition(
        model_decomposition,
        dl_train, dl_val, EPOCHS,
        LR, EPS, RHO,
        GROUP_SIZE)

    # Creazione e addestramento di un modello MBLD con batch scelto casualmente
    model_rnd = MLP(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
    model_rnd.to(device)
    avg_losses_rnd, losses_rnd, times_rnd, val_losses_rnd, val_accuracy_rnd = train_model_decomposition_rnd(model_rnd, dl_train, dl_val, EPOCHS,
                                                                                    LR, EPS, RHO, BATCH_SIZE)

    # Visualizzazione dei grafici
    plot_avg_loss(avg_losses_decomposition, avg_losses_rnd, 'MBLD', 'MBLD with random batch')
    plot_loss(losses_decomposition, losses_rnd, 'MBLD', 'MBLD with random batch')

    # Stampa dell'accuratezza e della loss dei due modelli
    print("Testing MBLD ...")
    test_loss_decomposition, test_accuracy_decomposition = evaluate_model(model_decomposition, dl_test, mode = "Test")
    print("\nTesting MBLD with random batch ...")
    test_loss_rnd, test_accuracy_rnd = evaluate_model(model_rnd, dl_test, mode = "Test")

    return test_loss_decomposition, test_accuracy_decomposition, test_loss_rnd, test_accuracy_rnd

# Metodo che confronta MBDL e MBLD con insiemi di GROUP_SIZE layer
def mbld_vs_sub(dl_train, dl_val, device):
    # Creazione e addestramento di un modello MBLD
    model_decomposition = MLP(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
    model_decomposition.to(device)
    avg_losses_decomposition, losses_decomposition, times_decomposition, val_losses_decomposition, val_accuracy_decomposition = train_model_decomposition(
        model_decomposition,
        dl_train, dl_val, EPOCHS,
        LR, EPS, RHO,
        GROUP_SIZE)

    # Creazione e addestramento di un modello MBLD con aggiornamento dei pesi relativi ad un sottoinsieme dei layers
    model_sub = MLP(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
    model_sub.to(device)
    avg_losses_sub, losses_sub, times_sub, val_losses_sub, val_accuracy_sub = train_model_decomposition_sub(model_sub, dl_train, dl_val, EPOCHS,
                                                                                    LR, EPS, RHO, GROUP_SIZE, BATCH_SIZE)

    # Visualizzazione dei grafici
    plot_avg_loss(avg_losses_decomposition, avg_losses_sub, 'MBLD', f'MBLD with groupsize = {GROUP_SIZE}')
    plot_loss(losses_decomposition, losses_sub, 'MBLD', f'MBLD with groupsize = {GROUP_SIZE}')

    # Stampa dell'accuratezza e della loss dei due modelli
    print("Testing MBLD ...")
    test_loss_decomposition, test_accuracy_decomposition = evaluate_model(model_decomposition, dl_test, mode = "Test")
    print(f"\nTesting MBLD with groupsize = {GROUP_SIZE}  ...")
    test_loss_sub, test_accuracy_sub = evaluate_model(model_sub, dl_test, mode = "Test")

    return test_loss_decomposition, test_accuracy_decomposition, test_loss_sub, test_accuracy_sub




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # Permette di eseguire l'addestramento su una GPU NVIDIA

    # Caricamento del dataset
    (Xs_train, ys_train, Xs_test, ys_test) = load_dataset()
    # Normalizzazione
    (Xs_train, ys_train, Xs_test, ys_test) = normalize_dataset(Xs_train, Xs_test, ys_train, ys_test, device)

    # Estrazione di un subset per il training
    train_size = 10000
    (Xs_train_ss, ys_train_ss) = subset_train(Xs_train, train_size)

    # Estrazione di un subset per la validazione
    val_size = 5000
    Xs_val, Xs_test = Xs_test[:val_size], Xs_test[val_size:]
    ys_val, ys_test = ys_test[:val_size], ys_test[val_size:]

    # Creazione di 64 batch
    (dl_train, dl_val, dl_test) = create_batch(BATCH_SIZE, Xs_train_ss, ys_train_ss, Xs_val, ys_val, Xs_test, ys_test)

    #best_params, best_score = grid_search(dl_train, dl_val, MLP)

    mbld_vs_minibatch(dl_train, dl_val, device)