# :snowflake: IceCube - Kaggle Challenge (2023)

My modest contribution to the IceCube Competition, hosted on Kaggle back in early 2023 ([link](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/overview)).

[![A mushroom-head robot](assets/img/detection.gif 'A detection example')](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/discussion/381166)

## :fire: Model

Before the competition, the best performing methods were Graph Neural Networks.

To practice my DS skills, I decided to implement the DynEdge architecture from scratch using **Pytorch** and **Torch-geometric**, using the official paper (https://arxiv.org/abs/2210.12194). At the time, this model had the best performing known architecture (scored MAE=1.018).

The official implementation of the model (GraphNet) can be found [here](https://github.com/graphnet-team/graphnet). It is pretty complex.

My implementation is slightly simpler and scores a decent MAE=1.07 while being trained on 10% of the dataset for 1 epoch, due to my limited resources.

*More details about my approach in the [doc](./doc/README.md) !*

[![Model architecture](./assets/img/model.png 'A detection example')](https://iopscience.iop.org/article/10.1088/1748-0221/17/11/P11003)

## :rocket: Run the code

If you consider running this code, I highly recommend to use Kaggle notebooks, so you don't have to download any data. The dataset ([here](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/data)) is massive (100Gi). Even standard Google Grive accounts are too small (15Gi) :sweat_smile:
