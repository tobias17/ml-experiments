# Cheat Sheet

An experiment in using the full transformer, encoder and decoder, to perform language modeling.

This specific experiment is using english wikipedia as reference material to the model while it is training with the hopes that we can see improved performance both in terms of accuracy but also memory requirements.

# Initial Setup

Create an `env.json` with the following fields:
```json
{
   "dataset_root": "/datasets/fineweb_cheat_sheet"
}
```
The above is the values I am using on my tinybox, but they will be used by the following code to know about your environment.

# Prepping the Dataset

**NOTE** The following dataset prep is for repeatability. A completed version will be uploaded and can be used instead.

## Download

The following will download all the necessary files.
```
python download_dataset.py
```

## Embed Wikipedia

Run the following script to split the en wiki dataset into chunks of text and create embeddings for them.
```
python embed_wikipedia.py
```

If you want to test that it works, run the following (note: this can be done before all the blobs are created).
```
python embed_wikipedia.py --run use
```

## Embed Fineweb

Run the following script to chunk the fineweb dataset and create embeddings for them.
```
python embed_fineweb.py
```

# Train the Model

TODO
