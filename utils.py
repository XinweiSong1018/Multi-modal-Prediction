import os.path
import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
import spacy
import logging
from collections import Counter
import csv
import pickle
import itertools
from sklearn.metrics import f1_score
from PIL import Image
import torchvision.transforms as transforms
import yaml

logger = logging.getLogger(__name__)

logger.info("----Loading Spacy----")
spacy_en = spacy.load('en_core_web_sm')


# Calculate F1: use scikit learn and use weighted and use
# data like this https://stackoverflow.com/questions/46732881/how-to-calculate-f1-score-for-multilabel-classification
# get statistics


def get_f1(y_pred, y_label):

    f1 = {"action_f1": f1_score(np.array(list(itertools.chain.from_iterable(y_pred[0]))),
                                list(itertools.chain.from_iterable(y_label[0])), average="weighted"),
          "object_f1": f1_score(np.array(list(itertools.chain.from_iterable(y_pred[1]))),
                                list(itertools.chain.from_iterable(y_label[1])), average="weighted"),
          "position_f1": f1_score(np.array(list(itertools.chain.from_iterable(y_pred[2]))),
                                  list(itertools.chain.from_iterable(y_label[2])), average="weighted")}

    return f1


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def numericalize(inputs, vocab=None, tokenize=False):
    # This should be 2 seperate functions
    # Create vocabs for train file
    if vocab == None:
        # check unique tokens
        counter = Counter()
        for i in inputs:
            if tokenize:
                counter.update(tokenizer(i))
            else:
                counter.update([i])

        # Create Vocab
        if tokenize:  # That is we are dealing with sentences.
            vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        else:
            vocab = {}

        vocab.update({j: i + len(vocab) for i, j in enumerate(counter)})

    # Convert tokens to numbers:
    numericalized_inputs = []
    for i in inputs:
        if tokenize:
            # Adding sos and eos tokens before and after tokenized string
            numericalized_inputs.append([vocab["<sos>"]]+[vocab[j] if j in vocab else vocab["<unk>"] for j in
                                         tokenizer(i)]+[vocab["<eos>"]])  # TODO: doing tokenization twice here
        else:
            numericalized_inputs.append(vocab[i])

    return numericalized_inputs, vocab


def collate_fn(batch,device, text_pad_value, audio_pad_value,audio_split_samples):
    """
    We use this function to pad the inputs so that they are of uniform length
    and convert them to tensor

    Note: This function would fail while using Iterable type dataset because
    while calculating max lengths the items in iterator will vanish.
    """
    max_audio_len = 0
    max_text_len = 0

    batch_size = len(batch)

    for audio_clip, transcript, _, _, _ in batch:
        if len(audio_clip) > max_audio_len:
            max_audio_len = len(audio_clip)
        if len(transcript) > max_text_len:
            max_text_len = len(transcript)

    # We have to pad the audio such that the audio length is divisible by audio_split_samples
    max_audio_len = (int(max_audio_len/audio_split_samples)+1)*audio_split_samples

    audio = torch.FloatTensor(batch_size, max_audio_len).fill_(audio_pad_value).to(device)
    text = torch.LongTensor(batch_size, max_text_len).fill_(text_pad_value).to(device)
    action = torch.LongTensor(batch_size).fill_(0).to(device)
    object_ = torch.LongTensor(batch_size).fill_(0).to(device)
    position = torch.LongTensor(batch_size).fill_(0).to(device)

    for i, (audio_clip, transcript, action_taken, object_chosen, position_chosen) in enumerate(batch):
        audio[i][:len(audio_clip)] = torch.tensor(audio_clip.tolist())
        text[i][:len(transcript)] = torch.tensor(transcript)
        action[i] = action_taken

        object_[i] = object_chosen
        position[i] = position_chosen

    return audio, text, action, object_, position


class Dataset:
    def __init__(self, audio, text, image1, image2, y):
        self.audio = audio
        self.text = text
        self.image1 = image1
        self.image2 = image2
        self.y = y
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize((256, 256))
        ])


    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        _, wav = wavfile.read(self.audio[item])
        i1 = Image.open(self.image1[item])
        i2 = Image.open(self.image2[item])
        return wav, self.text[item], self.transform(i1), self.transform(i2), self.y[item]


def load_csv(path, file_name):
    # Loads a csv and returns columns:
    # Header: audio text image1 image2 y
    with open(os.path.join(path, file_name)) as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)
        au, t, i1, i2, y = [], [], [], [], []
        for row in csv_reader:
            au.append(row[0])
            t.append(row[1])
            i1.append(row[2])
            i2.append(row[3])
            y.append(float(row[4]))
    return [au, t, i1, i2, y]


def get_Dataset_and_vocabs(path, train_file_name, valid_file_name):
    train_data = load_csv(path, train_file_name)
    test_data = load_csv(path, valid_file_name)

    vocabs = []  # to store all the vocabs
    # audio location need not to be numercalized
    # audio files will be loaded in Dataset.__getitem__()
    numericalized_train_data = train_data  # to store train data after converting string to ints
    numericalized_test_data = test_data  # to store test data after converting strings to ints



    a, vocab = numericalize(train_data[1], tokenize=True)
    b, _ = numericalize(test_data[1], vocab=vocab, tokenize=True)

    numericalized_train_data[1] = a
    numericalized_test_data[1] = b
    vocabs.append(vocab)

    train_dataset = Dataset(*numericalized_train_data)
    valid_dataset = Dataset(*numericalized_test_data)

    Vocab = {'text_vocab': vocabs[0]}

    logger.info(f"Transcript vocab size = {len(Vocab['text_vocab'])}")


    # dumping vocab
    with open(os.path.join(path, "vocab"), "wb") as f:
        pickle.dump(Vocab, f)

    return train_dataset, valid_dataset, Vocab


def get_Dataset_and_vocabs_for_eval(path, valid_file_name):
    test_data = load_csv(path, valid_file_name)

    with open(os.path.join(path, "vocab"), "rb") as f:
        Vocab = pickle.load(f)

    numericalized_test_data = [test_data[0], numericalize(test_data[1], vocab=Vocab['text_vocab']), \
                               test_data[2], test_data[3], test_data[4]]

    valid_dataset = Dataset(*numericalized_test_data)

    return valid_dataset, Vocab


def initialize_weights(m):
    # if hasattr(m, 'weight') and m.weight.dim() > 1:
    #     nn.init.xavier_uniform_(m.weight.data)
    for name, param in m.named_parameters():
        if not isinstance(m, nn.Embedding):
            nn.init.normal_(param.data, mean=0, std=0.01)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, train_iterator, optim, clip):
    model.train()

    epoch_loss = 0

    # Tracking accuracies
    accuracy = []

    # for f1
    y_pred = []
    y_true = []

    for i, batch in enumerate(train_iterator):
        # running batch
        train_result = model(*batch)

        optim.zero_grad()

        loss = train_result["loss"]
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optim.step()

        # Statistics
        epoch_loss += loss.item()
        y_pred.append([train_result["predicted_y"].tolist()])
        y_true.append([batch[4].tolist()])

        accuracy.append(sum(train_result["predicted_y"] == batch[4]) / len(batch[4]) * 100)

    y_pred = list(zip(*y_pred))
    y_true = list(zip(*y_true))

    epoch_f1 = get_f1(y_pred, y_true)


    return epoch_loss / len(
        train_iterator), (epoch_f1)


def evaluate(model, valid_iterator):
    model.eval()

    epoch_loss = 0

    # Tracking accuracies
    accuracy = []

    # for f1
    y_pred = []
    y_true = []

    with torch.no_grad():
        for i, batch in enumerate(valid_iterator):
            # running batch
            valid_result = model(*batch)

            loss = valid_result["loss"]

            # Statistics
            epoch_loss += loss.item()

            y_pred.append([valid_result["predicted_y"].tolist()])
            y_true.append([batch[4].tolist()])

            accuracy.append(sum(valid_result["predicted_y"] == batch[4]) / len(batch[4]) * 100)

    y_pred = list(zip(*y_pred))
    y_true = list(zip(*y_true))

    epoch_f1 = get_f1(y_pred, y_true)


    return epoch_loss / len(
        valid_iterator), (epoch_f1)


def add_to_writer(writer,epoch,train_loss,valid_loss,train_stats,valid_stats,config):
    writer.add_scalar("Train loss", train_loss, epoch)
    writer.add_scalar("Validation loss", valid_loss, epoch)
    writer.add_scalar("Train Action f1", train_stats[0]['action_f1'], epoch)
    writer.add_scalar("Train Object f1", train_stats[0]['object_f1'], epoch)
    writer.add_scalar("Train Position f1", train_stats[0]['position_f1'], epoch)
    writer.add_scalar("Train action accuracy", train_stats[1], epoch)
    writer.add_scalar("Train object accuracy", train_stats[2], epoch)
    writer.add_scalar("Train location accuracy", train_stats[3], epoch)
    writer.add_scalar("Valid Action f1", valid_stats[0]['action_f1'], epoch)
    writer.add_scalar("Valid Object f1", valid_stats[0]['object_f1'], epoch)
    writer.add_scalar("Valid Position f1", valid_stats[0]['position_f1'], epoch)
    writer.add_scalar("Valid action accuracy", valid_stats[1], epoch)
    writer.add_scalar("Valid object accuracy", valid_stats[2], epoch)
    writer.add_scalar("Valid location accuracy", valid_stats[3], epoch)


    writer.flush()
