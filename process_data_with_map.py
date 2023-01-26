"""Process video data into validation dataset"""
import time
from pathlib import Path
import datasets
from datasets import Dataset, Value, Array4D, Features

# Can try with either of these...
from pytorchvideo.data.encoded_video_pyav import EncodedVideoPyAV
from pytorchvideo.data.encoded_video_decord import EncodedVideoDecord

from io import BytesIO
from pytorchvideo.data.clip_sampling import make_clip_sampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Div255,
    Normalize,
    Permute,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import CenterCrop, Compose, RandomHorizontalFlip, RandomVerticalFlip

# 100 GB. Don't know if this really does anything
datasets.config.IN_MEMORY_MAX_SIZE = 1e+11


num_clip_frames = 16
short_side_size = 320
crop_height = 256

default_val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_clip_frames),
                    Div255(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ShortSideScale(size=short_side_size),
                    CenterCrop(size=crop_height),
                    Permute(dims=[1, 0, 2, 3]),
                ]
            ),
        ),
        RemoveKey(key="audio"),
    ]
)

clip_sampler = make_clip_sampler('uniform', 64 / 30)


def main(
    data_dir='./data/test/query',
    out_dir='test_query_processed',
    batch_size=32,
    num_proc=1,
    limit=None,
    writer_batch_size=16
):
    start = time.time()

    data_dir = Path(data_dir)
    video_paths = sorted(data_dir.glob('*.mp4'))
    video_ids= [x.stem for x in video_paths]
    video_paths = [str(x) for x in video_paths]

    if limit is not None:
        ds = Dataset.from_dict({'video_id': video_ids[:limit], 'video_path': video_paths[:limit]})
    else:
        ds = Dataset.from_dict({'video_id': video_ids, 'video_path': video_paths})

    num_examples = limit or len(video_paths)

    def split_video_clips(examples):
        examples_out = {'video_id': [], 'video_path': [], 'clip_index': [], 'clip_start': [], 'clip_end': [], 'video': []}

        for video_id, video_path in zip(examples['video_id'], examples['video_path']):
            video_bytes = BytesIO(Path(video_path).read_bytes())
            vid = EncodedVideoPyAV(video_path, video_name=video_path, decode_audio=False)

            next_clip_start_time = 0
            clip_sampler.reset()
            info_dict = {}
            while True:
                (
                    clip_start,
                    clip_end,
                    clip_index,
                    aug_index,
                    is_last_clip
                ) = clip_sampler(
                    next_clip_start_time, vid.duration, info_dict
                )

                clip = vid.get_clip(clip_start, clip_end)

                examples_out['video_id'].append(video_id)
                examples_out['video_path'].append(vid.name)
                examples_out['clip_index'].append(clip_index)
                examples_out['clip_start'].append(float(clip_start))
                examples_out['clip_end'].append(float(clip_end))

                video_out = default_val_transform(clip)['video']
                examples_out['video'].append(video_out.numpy())

                if is_last_clip:
                    break

                next_clip_start_time = clip_end

        return examples_out


    new_ds_features = Features(
        video_id=Value(dtype='string'),
        video_path=Value(dtype='string'),
        clip_index=Value(dtype='int64'),
        clip_start=Value(dtype='float64'),
        clip_end=Value(dtype='float64'),
        video=Array4D((16, 3, 256, 256), dtype='float32')
    )
    new_ds = ds.map(
        split_video_clips,
        batched=True,
        remove_columns=["video_id", "video_path"],
        features=new_ds_features,
        batch_size=batch_size,
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        keep_in_memory=False
    )

    end = time.time()

    print(f"\n\nThat took {end-start:.3f} seconds to process {num_examples} examples with batch size: {batch_size} and {num_proc} processes.\n\n")
    new_ds.save_to_disk(out_dir)
    return new_ds

if __name__ == '__main__':
    from fire import Fire
    Fire(main)