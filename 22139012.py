import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load dataset
print("\nLoad MNIST Database")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Done!")
x_train = np.reshape(x_train, (-1, 784)) / 255.0
x_test = np.reshape(x_test, (-1, 784)) / 255.0
y_train = np.eye(10)[y_train] #one-hot
y_test = np.eye(10)[y_test]

print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}")
print(f"y_test: {y_test.shape}")
print()

# Initialize hyper-parameters
lr = 0.05
epochs = 100
train_samples = x_train.shape[0]
test_samples = x_test.shape[0]
inputs_size = 784
h1_units = 512
h2_units = 128
num_classes = 10
miniBatch_size = 1000

# Initialize parameters
Wh1_init = np.random.uniform(low=-0.5, high=0.5, size=(h1_units, inputs_size)) 
bj_init = np.zeros((1, h1_units))                                          
Wh2_init = np.random.uniform(low=-0.5, high=0.5, size=(h2_units, h1_units))
bl_init = np.zeros((1, h2_units))                                            
Wo_init = np.random.uniform(low=-0.5, high=0.5, size=(num_classes, h2_units)) 
bk_init = np.zeros((1, num_classes))

print(f"Wh1_init: {Wh1_init.shape}")
print(f"bh1_init: {bj_init.shape}")
print(f"Wh2_init: {Wh2_init.shape}")
print(f"bh2_init: {bl_init.shape}")
print(f"Wo_init: {Wo_init.shape}")
print(f"bo_init: {bk_init.shape}")

# Activation functions
sigmoid = lambda x: 1. / (1. + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

ReLU = lambda x: np.maximum(0, x)

# Forward pass
def Forwardpass(X, Wh1, bj, Wh2, bl, Wo, bk):
    zh1 = np.matmul(X, Wh1.T) + bj
    a1 = ReLU(zh1)        
    zh2 = np.matmul(a1, Wh2.T) + bl 
    a2 = sigmoid(zh2)     
    z =np.matmul(a2, Wo.T) + bk
    o = softmax(z)         
    return zh1, a1, zh2, a2, o

# Accuracy
def AccTest(label, prediction):
    return np.mean(np.argmax(prediction, axis=1) == np.argmax(label, axis=1))


# Train function
def train(x_train, y_train, x_test, y_test, train_samples, batch_size, Wh1, bj, Wh2, bl, Wo, bk, epochs, lr):
    loss_history = []
    acc_history = []
    
    print("\n----- Neural Network Mini-Batch GD -----")
    print(f"Input Size: (1000, 784) {inputs_size*batch_size}")
    print("Hidden Layer 1:")
    print(f"\tActivation Shape: ({miniBatch_size}, {h1_units})")
    print(f"\tActivation Size: {miniBatch_size * h1_units}")
    print(f"\tParameters: {Wh1_init.size + bj_init.size}")
    print("Hidden Layer 2:")
    print(f"\tActivation Shape: ({miniBatch_size}, {h2_units})")
    print(f"\tActivation Size: {miniBatch_size * h2_units}")
    print(f"\tParameters: {Wh2_init.size + bl_init.size}")
    print("Output Layer:")
    print(f"\tActivation Shape: ({miniBatch_size}, {num_classes})")
    print(f"\tActivation Size: {miniBatch_size * num_classes}")
    print(f"\tParameters: {Wo_init.size + bk_init.size}")
    print(f"Total parameter: {Wh1_init.size + bj_init.size + Wh2_init.size + bl_init.size + Wo_init.size + bk_init.size}\n")
    
    for ep in range(1, epochs + 1):
        Stochastic_samples = np.arange(train_samples)
        np.random.shuffle(Stochastic_samples)
        epoch_loss = 0
        
        for ite in range(0, train_samples, batch_size):
            Batch_samples = Stochastic_samples[ite:ite + batch_size]
            x = x_train[Batch_samples, :]
            y = y_train[Batch_samples, :]
            
            # Forward pass
            zh1, a1, zh2, a2, o = Forwardpass(x, Wh1, bj, Wh2, bl, Wo, bk)
            
            # Compute loss
            batch_loss = -np.sum(np.multiply(y, np.log(o + 1e-9))) / batch_size
            epoch_loss += batch_loss
            
            # Backpropagation
            d = o - y
            dh2 = np.matmul(d, Wo)
            dh2s = np.multiply(np.multiply(dh2, a2), (1 - a2))
            dh1 = np.matmul(dh2s, Wh2)
            dh1s = np.multiply(dh1, (zh1 > 0))
            
            dWo = np.matmul(d.T, a2)
            dbo = np.sum(d, axis=0)
            dWh2 = np.matmul(dh2s.T, a1)
            dbh2 = np.sum(dh2s, axis=0)
            dWh1 = np.matmul(dh1s.T, x)
            dbh1 = np.sum(dh1s, axis=0)
            
            # Update parameters
            Wo -= lr * dWo / batch_size
            bk -= lr * dbo  / batch_size
            Wh2 -= lr * dWh2 / batch_size
            bl -= lr * dbh2  / batch_size
            Wh1 -= lr * dWh1 / batch_size
            bj -= lr * dbh1 / batch_size
            
        # Save loss
        loss_history.append(epoch_loss / (train_samples // batch_size))
        
        # Compute accuracy
        _, _, _, _, prediction = Forwardpass(x_test, Wh1, bj, Wh2, bl, Wo, bk)
        acc = AccTest(y_test, prediction)
        acc_history.append(acc)
        
        if ep % 10 == 0:
            print(f"Epoch {ep}/{epochs} - Training Loss: {loss_history[ep-1]:.4f} - Test Accuracy: {acc_history[ep-1]:.4f}")
    
    return Wh1, bj, Wh2, bl, Wo, bk, loss_history, acc_history

# Train
Wh1, bj, Wh2, bl, Wo, bk, loss_history, acc_history = train(
    x_train, y_train, x_test, y_test, train_samples, miniBatch_size,
    Wh1_init, bj_init, Wh2_init, bl_init, Wo_init, bk_init, epochs, lr)

#Evaluate
_, _, _, _, o = Forwardpass(x_test, Wh1, bj, Wh2, bl, Wo, bk)
accuracy = AccTest(y_test, o)*100
print(f"Accuracy: {accuracy:.3f}%")

# Visualize with plots
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(loss_history) + 1), loss_history, label='Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(acc_history) + 1), acc_history, label='Accuracy', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()