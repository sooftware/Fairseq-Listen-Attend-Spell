# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torchaudio
import random
import numpy as np
from . import data_utils
from fairseq.data import FairseqDataset
from .collaters import Seq2SeqCollater


class SpeechDataset(FairseqDataset):
    """
    A dataset representing speech and corresponding transcription.

    Args:
        audio_paths: (List[str]): A list of str with paths to audio files.
        audio_durations_ms (List[int]): A list of int containing the durations of audio files.
        targets (List[torch.LongTensor]): A list of LongTensors containing the indices of target transcriptions.
        target_dict (~fairseq.data.Dictionary): target vocabulary.
        ids (List[str]): A list of utterance IDs.
        num_mel_bins (int): Number of triangular mel-frequency bins (default: 80)
        frame_length (float): Frame length in milliseconds (default: 25.0)
        frame_shift (float): Frame shift in milliseconds (default: 10.0)
    """

    def __init__(
            self, audio_paths, audio_durations_ms, targets,
            target_dict, ids, speakers,
            num_mel_bins=80, frame_length=25.0, frame_shift=10.0,
            apply_spec_augment=True
    ):
        assert frame_length > 0
        assert frame_shift > 0
        assert all(x > frame_length for x in audio_durations_ms)

        self.frame_sizes = [int(1 + (d - frame_length) / frame_shift) for d in audio_durations_ms]

        assert len(audio_paths) > 0
        assert len(audio_paths) == len(audio_durations_ms)
        assert len(audio_paths) == len(targets)
        assert len(audio_paths) == len(ids)

        self.audio_paths = audio_paths
        self.target_dict = target_dict
        self.targets = targets
        self.ids = ids
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.speakers = speakers
        self.apply_spec_augment = apply_spec_augment

        self.s2s_collater = Seq2SeqCollater(
            feature_index=0,
            label_index=1,
            pad_index=self.target_dict.pad(),
            eos_index=self.target_dict.eos(),
            move_eos_to_beginning=True
        )

    def __getitem__(self, index):
        target = self.targets[index] if self.targets is not None else None
        path = self.audio_paths[index]

        if not os.path.exists(path):
            raise FileNotFoundError("Audio file not found: {}".format(path))

        signal, sample_rate = torchaudio.load_wav(path)
        output = torchaudio.compliance.kaldi.fbank(
            signal,
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift
        )
        output = data_utils.apply_mv_norm(output)

        # if self.apply_spec_augment:
        #     if self.spec_augment_flags[index]:
        #         output = self.spec_augment(output)

        return {"id": index, "data": [output.detach(), target]}

    def __len__(self):
        return len(self.audio_paths)

    def collater(self, samples):
        """
        Merge a list of samples to form a mini-batch.

        Args:
            samples (List[int]): sample indices to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        return self.s2s_collater.collate(samples)

    def prepare_spec_augment(self, dataset_size):
        for idx in range(dataset_size):
            self.frame_sizes.append(self.frame_sizes[idx])
            self.audio_paths.append(self.audio_paths[idx])
            self.targets.append(self.targets[idx])
            self.speakers.append(self.speakers[idx])
            self.spec_augment_flags.append(True)

    def shuffle(self):
        tmp = list(zip(self.audio_paths, self.targets, self.spec_augment_flags, self.frame_sizes))
        random.shuffle(tmp)
        self.audio_paths, self.targets, self.spec_augment_flags, self.frame_sizes = zip(*tmp)

    def num_tokens(self, index):
        return self.frame_sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.frame_sizes[index],
            len(self.targets[index]) if self.targets is not None else 0
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self))
