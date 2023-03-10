# Speech De-warping
PyTorch implementation of the paper "Unsupervised Pre-training for Data-Efficient Text-to-Speech on Low Resource Languages", ICASSP 2023.
The repository is currently under construction.

> **Abstract:** 
> Neural text-to-speech (TTS) models can synthesize natural human speech when trained on large amounts of transcribed speech.
> However, collecting such large-scale transcribed data is expensive.
> This paper proposes an unsupervised pre-training method for a sequence-to-sequence TTS model by leveraging large untranscribed speech data.
> With our pre-training, we can remarkably reduce the amount of paired transcribed data required to train the model for the target downstream TTS task. 
> The main idea is to pre-train the model to reconstruct de-warped mel-spectrograms from warped ones, which may allow the model to learn proper temporal assignment relation between input and output sequences.
> In addition, we propose a data augmentation method that further improves the data efficiency in fine-tuning.
> We empirically demonstrate the effectiveness of our proposed method in low-resource language scenarios, achieving outstanding performance compared to competing methods.
