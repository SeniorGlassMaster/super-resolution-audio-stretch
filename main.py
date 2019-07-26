import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from librosa.core import load
import matplotlib.pyplot as plt
import load_data
from models import *
from parameters import *
from export_audio import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("Loading audio files...")
    input_audio, sr = load("./midi_renders/fugue_1_plucks.wav")
    target_audio, sr_target = load("./midi_renders/fugue_1_plucks_slow.wav")
    assert sr == sr_target, "Sample rate of input and target audio not equal."

    if MODEL == 'pre':
        input_audio, target_audio = load_data.pre_model_prepare(input_audio, target_audio)
        model = Pre_Upscale_Model().to(device)
    elif MODEL == 'post':
        input_audio, target_audio = load_data.post_model_prepare(input_audio, target_audio)
        model = Post_Upscale_Model().to(device)
    elif MODEL == 'pre_s':
        input_data, target_data = load_data.pre_model_s_prepare(input_audio, target_audio, sr)
        model = Pre_Upscale_Spectrogram_Model().to(device)

    model = model.double()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if LOAD_MODEL is True:
        model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location='cpu'))
    else:
        print("Training model...")

        if MODEL != 'pre_s':
            loss_plot = None
            line1 = None
            fig = None

            if LIVE_GRAPH:
                loss_plot = []
                x_plot = []
                plt.ion()
                fig = plt.figure()
                ax = fig.add_subplot(111)
                line1, = ax.plot(x_plot, loss_plot, '-', color='orange', linewidth=1.0)

            for epoch in range(1, NUM_EPOCHS + 1):
                train_model(model, input_audio, target_audio, optimizer, epoch, loss_plot, line1, fig)

                cur_save_path = SAVE_PATH + "_e" + str(epoch) + ".pth"
                torch.save(model.state_dict(), cur_save_path)
                print("   Saved model to " + cur_save_path)
        else:
            train_model_s(model, input_data, target_data, optimizer)

    if MODEL != 'pre_s':
        test_result = test_model(model, input_audio, target_audio)
        render_audio(test_result, EXPORT_PATH, sr)
    else:
        test_result = test_model(model, input_data, target_data)
        render_audio_s(test_result, EXPORT_PATH, sr)

def train_model(model, input_data, target_data, optimizer, epoch, loss_plot, line, fig):
    model.train()
    loss = nn.MSELoss()
    # loss = nn.CTCLoss()
    # loss = nn.L1Loss()
    total_loss = 0
    sample_size = input_data.shape[0]
    input_data = input_data * 1e4
    target_data = target_data * 1e4
    for i in range(sample_size):
        input_window = torch.tensor(np.array([[input_data[i]]])).to(device)
        target_window = torch.tensor(np.array([[target_data[i]]])).to(device)
        optimizer.zero_grad()
        output = model(input_window.double())
        # train_loss = loss(output, target_window.double())
        train_loss = loss(output, target_window.double())
        train_loss.backward()
        total_loss += train_loss.sum()

        optimizer.step()
        progress = 100 * ((i + 1) / sample_size)
        if progress != 100 and (int(progress * 10) % 7) == 0:
            progress_str = str(progress)[:4]
            print('   Epoch {}: {}% complete   '.format(epoch, progress_str), end='\r')
        elif progress == 100:
            print('   Epoch {}: 100% complete   '.format(epoch))

        if LIVE_GRAPH:
            loss_plot.append(train_loss.sum().item())
            if i % 10 == 0:
                fig.canvas.start_event_loop(0.0001)
                line.set_xdata(list(range(len(loss_plot))))
                line.set_ydata(loss_plot)
                plt.axis([0,len(loss_plot),1e-5,max(loss_plot)])
                plt.yscale("log")
                fig.canvas.draw()
                fig.canvas.flush_events()

    print('   Train Epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss))

def test_model(model, input_data, target_data):
    model.eval()
    loss = nn.MSELoss()
    test_loss = 0
    stitched_audio = []
    with torch.no_grad():
        for i in range(input_data.shape[0]):
            input_window = torch.tensor([[input_data[i]]]).to(device)
            target_window = torch.tensor([[target_data[i]]]).to(device)
            output = model(input_window.double())
            stitched_audio.append(output[0,0].numpy())
            test_loss += loss(output, target_window.double()).mean()

    test_loss /= input_data.shape[0]
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))

    stitched_audio = np.array(stitched_audio)
    # stitched_audio = np.concatenate(stitched_audio, axis=None)
    return stitched_audio

def test_model_s(model, input_data, target_data):
    model.eval()
    loss = nn.MSELoss()
    with torch.no_grad():
        output = model(input_data.double())
        test_loss = loss(output, target_data.double()).sum()

    print('\nTest set --- loss sum: {:.4f}'.format(test_loss))
    return output

def train_model_s(model, input_data, target_data, optimizer):
    model.train()
    loss = nn.MSELoss()
    # loss = nn.CTCLoss()
    # loss = nn.L1Loss()

    if LIVE_GRAPH:
        loss_plot = []
        x_plot = []
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line, = ax.plot(x_plot, loss_plot, '-', color='orange', linewidth=1.0)

    for epoch in range(1, NUM_EPOCHS + 1):
        optimizer.zero_grad()
        output = model(input_data)
        train_loss = loss(output, target_data)
        train_loss.backward()
        optimizer.step()

        # progress = 100 * ((i + 1) / sample_size)
        # if progress != 100 and (int(progress * 10) % 7) == 0:
        #     progress_str = str(progress)[:4]
        #     print('   Epoch {}: {}% complete   '.format(epoch, progress_str), end='\r')
        # elif progress == 100:
        #     print('   Epoch {}: 100% complete   '.format(epoch))
            
        cur_save_path = SAVE_PATH + "_e" + str(epoch) + ".pth"
        torch.save(model.state_dict(), cur_save_path)
        print("   Saved model to " + cur_save_path)

        if LIVE_GRAPH:
            loss_plot.append(train_loss.sum().item())
            fig.canvas.start_event_loop(0.0001)
            line.set_xdata(list(range(len(loss_plot))))
            line.set_ydata(loss_plot)
            plt.axis([0,len(loss_plot),1e-5,max(loss_plot)])
            plt.yscale("log")
            fig.canvas.draw()
            fig.canvas.flush_events()

        print('   Train Epoch: {} \tLoss: {:.6f}'.format(epoch, train_loss.sum()))

if __name__ == "__main__":
    main()