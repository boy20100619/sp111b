# -*- coding: utf-8 -*-
"""whisper.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DSQeE5JdF8yuK9fA77_9fj7D8B4-Usdk

程式碼修改來源自Github上[OpenAI的whisper](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)專案，使用colab範例進行測試，本人(梁齊恆)進行部分添加修改和中文註解

 官方[LICENSE](https://github.com/openai/whisper/blob/main/LICENSE)

# Installing Whisper /下載whisper專案

The commands below will install the Python packages needed to use Whisper models and evaluate the transcription results.

安裝使用 Whisper 模型所需的Python包和套件，下面要使用資料集測試準確率
"""

! pip install git+https://github.com/openai/whisper.git
! pip install jiwer

"""# Loading the LibriSpeech dataset
The following will load the test-clean split of the LibriSpeech corpus using torchaudio.

## /加載librispeech資料集
LibriSpeech 是一個包含約 1000 小時閱讀英語語音的語音資料庫，torchaudio主要處理音頻

"""

import os
import numpy as np

try:
    import tensorflow  # 避免colab的兼容性問題
except ImportError:
    pass

import torch
import pandas as pd
import whisper
import torchaudio

from tqdm.notebook import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, split="test-clean", device=DEVICE):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset) #計算長度

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device) #將音訊進行修剪與填充，使其長度為30秒
        mel = whisper.log_mel_spectrogram(audio)

        return (mel, text) #計算音訊的Mel頻譜圖。最後返回一個元組(mel, text)，其中mel是梅爾頻譜圖，text是音訊對應的文字。

dataset = LibriSpeech("test-clean")
loader = torch.utils.data.DataLoader(dataset, batch_size=16)

"""# Running inference on the dataset using a base Whisper model

The following will take a few minutes to transcribe all utterances in the dataset.

## /使用基礎 whisper模型進行資料推論
"""

#@markdown 語言分成English-only model和Multilingual model，可用模型由小到大分為`tiny`、`base`、`small`、`medium`、`large`，English-only沒有large模型
model = whisper.load_model("base.en") #選定English-only的base模型
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

# predict without timestamps for short-form transcription
options = whisper.DecodingOptions(language="en", without_timestamps=True)

hypotheses = []
references = []

for mels, texts in tqdm(loader):
    results = model.decode(mels, options)
    hypotheses.extend([result.text for result in results])
    references.extend(texts)

data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
data

"""# Calculating the word error rate

Now, we use our English normalizer implementation to standardize the transcription and calculate the WER.

## /計算單詞的錯誤率
jiwer是 automatic speech recognition system 自動語音辨識系統，主要能分析:
1.   word error rate (WER)
2.   match error rate (MER)
3.   word information lost (WIL)
4.   word information preserved (WIP)
5.   character error rate (CER)


"""

import jiwer
from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer() #英文文本正規化處理

data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
data["reference_clean"] = [normalizer(text) for text in data["reference"]]
data

wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))

print(f"Word Error Rate: {wer * 100:.2f} %") #錯誤率

"""#
___
## 補充whisper在python套件中直接使用

"""

!git clone https://github.com/boy20100619/test.git
import whisper

model = whisper.load_model("base")
result = model.transcribe("./test/scottish-accent.wav") # model.transcribe("你的檔案路徑")
print("-------------------------------","\n")
print(result["text"])

import whisper

model = whisper.load_model("base")

#load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("./test/nahida_0.wav") # whisper.load_audio("你的檔案路徑")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
# 生成梅爾頻譜圖和切換模型
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language 檢測語言
_, probs = model.detect_language(mel)
print(f"Detected language is: {max(probs, key=probs.get)}")

# decode the audio 解碼音頻
#options = whisper.DecodingOptions()  #預設
options = whisper.DecodingOptions(fp16 = False) # 如果不支援 fp16,以 fp32 取代,須改為 False
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)

#@markdown 透過google colab直接上傳音檔測試whisper語音轉文字，可以用 .wav檔和 .mp3檔
from google.colab import files
import whisper

# 上傳音檔並檢測檔名
filename = ""
uploaded = files.upload()
for nametest in uploaded.keys():
  if ".wav" in nametest or ".mp3" in nametest:
    filename = nametest

if filename != "":
  audio = whisper.load_audio(filename)
  audio = whisper.pad_or_trim(audio)

  mel = whisper.log_mel_spectrogram(audio).to(model.device)
  _, probs = model.detect_language(mel)
  print("\n---",f"檢測語言為: {max(probs, key=probs.get)}")
  options = whisper.DecodingOptions(fp16 = False)
  result = whisper.decode(model, mel, options)

  print(result.text)
  # 保存文字檔
  with open(f"output_{filename[:-4]}.txt",'w') as f:
    f.write(result.text)