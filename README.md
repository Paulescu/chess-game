## Let's build a Chess game using LLMs running privately on your phone/computer/microwave

The idea is simple.
We fine-tune a small (350M parameter) LLM to play chess as the great Magnus Carlsen.
Then we deploy this in an iOS game you can play with.

## Steps

The steps to create this project are:

1. Download raw data with Magnus 7k games
2. Generate instruction dataset (and validate it!)
3. Fine-tune LFM2-350M to imitate Magnus (aka LFM2-350M-MagnusInstruct)
4. Evaluate the model by playing against it
5. Bundle the model with `leap-bundle`
6. Create a simple iOS game and embed our fine-tuned LLM using the `LeapSDK` for iOS.

Steps 1 to 5 are implemented inside the `fine-tune` directory.
The iOS app is inside `LeapChatApp`

Let me go over each of these steps, step by step :-)


## 1. Download raw data

We take 7k historical games played by the great Magnus Carlsen from [this URL](https://www.pgnmentor.com/files.html#players) and generate tuples containing

- Current game state expressed in FEN
- Previous 5 moves
- Next move picked by Magnus

Warning:
An LLM that predicted the next move well in the test (even validation set) is not necessarily
a good chess player.

As a matter of fact, playing random against this LLM is actually a great way to win against it.
Playing random exposes the LLM to a high-number of out-of-sample situation it has never observed during training (remember, chess is a highly-dimenstional state space game) and the model is not really trained
to reason like good chess players do, combining long-term strategic play and short-term tactical play.

Because of this, training a reasoning LLMs using next moves and expert human strategy and tactis
would work better. See [this paper for further reference](https://arxiv.org/html/2411.06655v2).

## 2. Generate instruction dataset

## 3. Fine-tune LFM2-350M to imitate Magnus

## 4. Evaluate the AI player we just created (spoiler alert: it sucks! more on this below)

## 5. Bundle the model with the instruction dataset

## 6. Create iOS app with LFM2-350M-MagnusInstruct

First I decided to bootstrap a minimal chat app that loads the model and allows you to chat with it.
For that I used the `LeapChatExample` from the `LeapSDK-Examples` repository.
You can find it [here](https://leap.liquid.ai/docs/edge-sdk/ios/ios-quick-start-guide#complete-example)

### Steps to create the app

- Create a new project with xcode.
- Create a `Resources` folder and add the model bundle `LFM2-350M-MagnusInstruct.bundle`.
- Add the `LeapSDK` package to the project as explained [here](LFM2-350M-MagnusInstruct).


ChessboardKit for a nice UI:
https://github.com/rohanrhu/ChessboardKit/tree/main?tab=readme-ov-file#quick-start