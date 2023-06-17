# Whisper 語音轉文字模型
### 聲明
資料參考來自OpenAI在github上的專案[Whisper](https://github.com/openai/whisper)，並在官方colab範例中進行註解和修改

其他參考資料：
* [OpenAI 免費開源語音辨識系統-- Whisper 安裝簡介及原理](https://ithelp.ithome.com.tw/articles/10311957)
* [Introducing Whisper](https://openai.com/research/whisper)
* [torchaudio.datasets - PyTorch](https://pytorch.org/audio/stable/datasets.html)

[LICENSE](/finalexam/LICENSE)

建議[在google colab中運行](https://colab.research.google.com/drive/1DSQeE5JdF8yuK9fA77_9fj7D8B4-Usdk?usp=sharing)查看python程式碼
___
## 基本介紹
### 一、原理與架構
Whisper 是 OpenAI 提供的一種開源的自動語音辨識(Automatic Speech Recognition，ASR)的神經網路模型，用來執行多語言語音辨識(multilingual speech recognition)與語言翻譯(speech translation)的功能。能夠將各種語言的語音轉錄成文字(speech to text)，甚至可以處理較差的音頻品質或過多的背景雜訊。

Transformer sequence-to-sequence model是將 Multitask training data (多工訓練資料) 先轉換成梅爾頻譜圖(Mel Spectrogram)再透過大量多層感知機(Multilayer perceptron,MLP)組成Encoder和 Decoder 兩個 RNN 以進行編碼和解碼的方式學習，多任務訓練格式由解碼器將一系列tokens作標記和分類，使得替代傳統語音處理流程中的多個階段。最終可以實現多語言語音識別與翻譯。

1.  何謂梅爾頻譜？ 其實它就是在語音處理中常常用到的頻譜，在語音分類中把信號變成圖片，然後用分類圖片的算法 (例如: CNN) 來分類語音。
2.  頻譜經由 CNN (2xConv1D + GELU) 的方法 training，再經由正弦波位置編碼(Sinusoidal position encoder)，簡單講就是在 transformer 架構中加入一些關於 token 位置的信息，其演算法使用了不同頻率的正弦函數来作為位置編碼，所以叫作正弦波位置編碼。
3.  接著進入 Tranformer Encoder 的過程，然後再經由 cross attention 到 transformer decoder 輸出。在訓練過程中我們可以得知 Multitask training format 中 token 的信息，與訓練好的模型來預測下一個 token。這也意味著可以經由模型來輸出講者的說話模式或內容。
4.  最下方為 Multitask training format，主要分為三種類型的 token： special tokens、text tokens、timestamp tokens；一開始先將原本輸入的語音內容分為兩類：有語音與沒有語音；沒有語音則經由語音活性檢測 (Voice activity detection，VAD)再度確認，或是將其處理掉；若有語音則分為英語系和非英語系，兩種都可由下列兩種方式錄製或翻譯：
    * 需要時間校準(Time-aligned Transcription): 於 text token 之間，插入 begin time 和 end time。
    * 不需時間校準 (Text-only Transcription): 只有 text tokens，但它可以允許 dataset fine tune。
![approach](/finalexam/picture/approach.png)

### 二、可用模型與語言
官方提供5種尺寸的模型可供使用，以及所需使用的VRAM和相對速度，其中只有純英文模型沒有`large`尺寸。

並且提到`.en`模型對於只有英文的場合效果更佳，但同時`small.en`和`medium.en`的差異已經不太顯著了。

![models](/finalexam/picture/models.png)

whisper轉換文字的效率會跟使用的語言有差，下圖則是官方提供`large-v2`的多語言模型的WER(Word Error Rate,文字錯誤率)，數字越小代表該語言效果越好，其中錯誤率最低的是西班牙文，只有`3%`的錯誤率，而中文則約有`14.7%`的錯誤率。

![wer](/finalexam/picture/wer.png)
___
## 程式碼
在colab範例中，使用了Pytorch中的database：LibriSpeech作為測試資料和torchaudio套件進行音頻處理、以及jiwer來檢測模型轉換後的WER。

1. 安裝使用 Whisper 模型所需的Python套件和專案資料
```
pip install git+https://github.com/openai/whisper.git
pip install jiwer
```
2. 導入torchaudio和LibriSpeech，打包和處理音頻的資料
```
import os
import numpy as np

try:
    import tensorflow  
except ImportError:
    pass

import torch
import pandas as pd
import whisper
import torchaudio
from tqdm.notebook import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```
使用`whisper.pad_or_trim`方法進行修剪與填充，使其長度為30秒，並將結果轉移到指定的device。`whisper.log_mel_spectrogram`方法計算音訊的梅爾頻譜圖。最後返回一個元組(mel, text)，其中mel是Mel頻譜圖，text是音訊對應的文字。
```
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
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)

        return (mel, text)
```
```
dataset = LibriSpeech("test-clean")
loader = torch.utils.data.DataLoader(dataset, batch_size=16)
```
3. 載入whisper模型，並預判沒有timestamps的短語音
```
model = whisper.load_model("base.en")
options = whisper.DecodingOptions(language="en", without_timestamps=True)
```
對每個批次的梅爾頻譜圖進行解碼，並將解碼結果的文本和原始的參考文本分別存儲到hypotheses和references列表中。
```
hypotheses = []
references = []

for mels, texts in tqdm(loader):
    results = model.decode(mels, options)
    hypotheses.extend([result.text for result in results])
    references.extend(texts)
```
4. 導入jiwer和EnglishTextNormalizer，用於對英文文本進行正規化和預處，以便後續使用jiwer進行文本錯誤率計算。
```
import jiwer
from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()
data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
data["reference_clean"] = [normalizer(text) for text in data["reference"]]

wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
print(f"Word Error Rate: {wer * 100:.2f} %") 
```
5. 結果

![result](/finalexam/picture/result.png)

## 補充應用
直接將音檔轉換成文字輸出
```
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])
```
或是
```
import whisper

model = whisper.load_model("base")

#load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("你的檔案路徑")
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
```
結果：

![result2](/finalexam/picture/result2.png)

甚至申請一組OpenAI的API key，可以直接進行 OpenAI 語音辨識，不需要額外安裝 Whisper。
```
pip install openai
```
```
import os
import openai
openai.organization = "org-iLO9ZtJh7FCEufYHGQNle6ur"
openai.api_key = os.getenv("OPENAI_API_KEY")
```
