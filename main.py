import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from librosa.core import load
from librosa.output import write_wav
import load_data
from model import NN_Model

def train_model(model, input_data, target_data, optimizer, epoch):
    model.train()
    loss = nn.MSELoss()
    total_loss = 0
    for i in range(input_data.size):
        input_window = torch.tensor(np.array([[input_data[i]]]))
        target_window = torch.tensor(np.array([[target_data[i]]]))
        optimizer.zero_grad()
        output = model(input_window)
        train_loss = loss(output, target_window.double())
        train_loss.backward()
        total_loss += train_loss.mean()
        optimizer.step()

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss))

def test_model(model, input_data, target_data):
    model.eval()
    loss = nn.MSELoss()
    test_loss = 0
    stitched_audio = []
    with torch.no_grad():
        for i in range(input_data.size):
            input_window = torch.tensor([[input_data[i]]])
            target_window = torch.tensor([[target_data[i]]])
            output = model(input_window.double())
            stitched_audio.append(output[0,0].numpy())
            test_loss += loss(output, target_window.double()).mean()
    
    test_loss /= input_data.size
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    
    stitched_audio = np.array(stitched_audio)
    stitched_audio = np.concatenate(stitched_audio, axis=None)
    return stitched_audio

def main():
    WINDOW_SIZE = 2000
    NUM_EPOCHS = 8
    LEARNING_RATE = 1e-2
    input_audio, sr = load("./midi_renders/fugue_1_plucks.wav")
    target_audio, sr = load("./midi_renders/fugue_1_plucks_slow.wav")
    input_audio = load_data.preprocess_input_data(input_audio, WINDOW_SIZE)
    target_audio = load_data.preprocess_target_data(target_audio, WINDOW_SIZE)
    assert(input_audio.size == target_audio.size)

    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    model = NN_Model().to(device)
    model = model.double()
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_model(model, input_audio, target_audio, optimizer, epoch)
    
    input_audio, sr = load("./midi_renders/fugue_2_plucks.wav")
    target_audio, sr = load("./midi_renders/fugue_2_plucks_slow.wav")
    input_audio = load_data.preprocess_input_data(input_audio, WINDOW_SIZE)
    target_audio = load_data.preprocess_target_data(target_audio, WINDOW_SIZE)
    test_result = test_model(model, input_audio, target_audio)
    write_wav("./output/test_result.wav", test_result, sr)

if __name__ == "__main__":
    main()