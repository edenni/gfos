import torch


def create_pack_collator(max_seq: int):
    """Sequence length of a batch could be max_seq or the length of the longest sequence in the batch."""

    def packed_collate(batch):
        """
        Puts data, and lengths into a packed_padded_sequence then returns
        the packed_padded_sequence and the labels. Set use_lengths to True
        to use this collate function.

        Args:
            batch: list of [data, target].

        Output:
            packed_batch: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
            labels: (Tensor), labels from the file names of the wav.
        """
        data, labels, _ = zip(
            *[
                (a, b, a.shape[0])
                for (a, b) in sorted(
                    batch, key=lambda x: x[0].shape[0], reverse=True
                )
            ]
        )

        # Trim to num_seq
        data = [s[:max_seq] for s in data]

        labels = torch.stack(labels, 0)

        packed_batch = torch.nn.utils.rnn.pack_sequence(data)
        return packed_batch, labels

    return packed_collate


def create_pad_collator(
    max_seq: int, padding_value: float = 0.0, truncate_right: bool = False
):
    """Sequence length of a batch could be max_seq or the length of the longest sequence in the batch."""

    def pad_collate(batch):
        data, labels = zip(*batch)

        # Trim to num_seq
        if truncate_right:
            data = [s[-max_seq:] for s in data]
        else:
            data = [s[:max_seq] for s in data]

        labels = torch.stack(labels, 0)

        packed_batch = torch.nn.utils.rnn.pad_sequence(
            data, batch_first=True, padding_value=padding_value
        )
        return packed_batch, labels

    return pad_collate
