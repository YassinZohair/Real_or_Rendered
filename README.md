# Real or Rendered?

Detecting AI-generated images that generalize to unseen generators and survive real-world processing.
Course project for Advanced Machine Learning (CSCE 4604), AUC. Team: Yassin Zohair, Fady Morcos, Karim Atalla.

Baseline: UnivFD (Ojha et al., CVPR 2023), a frozen CLIP ViT-L/14 image encoder with one trainable linear layer.
Training data: GenImage, Stable Diffusion v1.4 subset. Test data: GenImage test splits for all eight generators.

## Running

Open `notebooks/baseline.ipynb` in Google Colab on a GPU runtime and run the cells in order.

| Step | Command | What it does |
|---|---|---|
| Authors' evaluation | `bash scripts/authors_eval.sh` | Runs the UnivFD repo unmodified on its released diffusion test set |
| Data | `python -m ror.data` | Downloads the GenImage shards from the Hugging Face mirror |
| Features | `python -m ror.features` | Caches CLIP features for the training sample and the test sample |
| Check | `python -m ror.check_faithful` | Compares the cached pipeline with the repo's model end to end |
| Baseline | `python -m ror.baseline` | Evaluates the released layer and our retrained layer per generator |
