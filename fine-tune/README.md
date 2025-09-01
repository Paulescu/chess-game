## Fine tuning LFM2-350M to play chess

Steps

1. Generate a high-quality (instruction, output) dataset.
2. Fine-tune LFM-350M with this dataset.
3. Validate the model is good enough, for example playing against its current production version and observing win rates.
4. Push the model to the HuggingFace model registry if the model is "good enough.|

    What does "good enough" mean?
    In this case, use the model you just trained to play against a previous version of it,
    for example the one that is currently deployed in your iOS app.
    If the win rate is above X, then push, otherwise back to the lab and try to build
    something better.

5. Bundle the model with `leap-bundle`
6. Load into the iOS app with the Leap Edge SDK.

## TODOs

- 