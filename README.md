# VSC Load With HF Datasets

Machine I'm on has 126G RAM.

## First download the data...

You'll need aws CLI installed if you don't already. Here's how on linux:

```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

Then run the download script

```
bash download.sh
```

## Install deps

I used new conda env with python 3.8

```
pip install -r requirements.txt
```

## Process the Data

#### With map (ends up freezing)

```
python process_data_with_map.py --batch_size 2 --writer_batch_size 2 --num_proc 4
```

#### With generator (works)

```
python process_data_from_generator.py --num_proc 4 --num_shards 8 --writer_batch_size 32
```