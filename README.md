## Let's build a Chess game using LLMs running privately on your phone, computer or microwave (if that is your thing)

### Table of contents

- [What is this repo about?](#what-is-this-repo-about)
- [Why a game?](#why-a-game)
- [Why local and small LLMs are the future (and present)?](#why-local-and-small-llms-are-the-future-and-present)
- [Steps](#steps)
- [1. Download chess moves dataset](#1-download-chess-moves-dataset)
- [2. Generate instruction dataset](#2-generate-instruction-dataset)
- [3. Fine-tune LFM2-350M to imitate Magnus Carlsen](#3-fine-tune-lfm2-350m-to-imitate-magnus)
- [4. Evaluate LFM2-350M-MagnusInstruct on real game play](#4-evaluate-lfm2-350m-magnusinstruct-on-real-game-play)
- [5. Bundle the model with the Leap Model Bundling CLI](#5-bundle-the-model-with-the-leap-model-bundling-cli)
- [6. Embed the model into an iOS app with the Leap Edge SDK](#6-create-ios-app-with-lfm2-350m-magnusinstruct)


## What is this repo about?

In this repository you will learn to fine-tune a small LLM to play chess and deploy it
locally in an iOS app.

We will use the `LiquidAI` family of open-weight `LFM2` small-frontier models (less than 1B parameters)
to embed super-specialized task-specific intelligence on the edge using the `LeapSDK` for iOS.

Now, before getting into the technical details (aka the HOWs), let me tell you the WHATs and WHYs.


## Why a game?
   
Mobile games are all about immersing the player in a virtual world, and customizing the
experience to the player to increase engagement and retention.

Large Language Models on the other hand, are great at generating text, images or audio dynamically,
based on the player's actions and the game state.

This makes LLMs perfect for mobile games... Well, there is one catch.

If the LLMs need to run on remote servers, the whole thing becomes super-expensive
and super-slow.

So it is not worth it. Unless you deploy the LLM on the client device.


## Why local and small LLMs are the future (and present)?

Let me tell you a story. It will be short, don't worry.

I used to work as a data scientist and ML engineer at Nordeus, creator and producer of one of the most successful mobile games of all time: "Top Eleven".
The game (like all games out there) is designed to immerse the player in a virtual world.
In this case a world where you (the player) are a soccer manager, and you have to

- train your squad
- compete against other managers in a live match.
- make player transfers
- etc.

Every day, several million players around the world were logging into the game, looking for their
daily dose of soccer.

Average playtime per day was something around 15 minutes.

So, can you imagine how many request and tokens flying-in-and-out of your LLM server would you need
to generate to keep several million players happy for 15 minutes?

I haven't calculated it, but I can tell you that it would be a lot. Too much to make LLMs running on remote servers worth it.

Hence, the only cost-effective way to use LLMs in this context is to specialize them
for the task we want them to solve (e.g. real time soccer gama narration) and make them
small enough so they can run on the player phone.

This is what makes local and small LLMs the future (and present) for mobile games.


## What if you don't care about games?

In this project we build a chess game, but the need for local and private LLMs deployments
is relevant for many other problems, especially when you need to handle sensitive data or work
reliably offline.

Imagine for example financial apps, insurance apps, health care apps, etc.
In these cases, the data is sensitive and you don't want to (or even can't) send it to a remote server.

Or self-driving cars and robots, that need to work reliably offline. You don't want your car
to start having problems when you are driving in the middle of nowhere.

In any case, the need for local and private LLMs deployments is becoming more and more relevant.

Because the future of LLMs is not super-expensive GPT-524 running on a data-center the size
of North Dakota.

The future of LLMs is high-quality-high-performance-low-latency-task-specific-low-cost-private-data
running on your phone, computer, robot, watch, tablet, raspberry pi or microwave, if that is your thing.
 

## Steps  

Let me now walk you over the main steps I followed (and you can follow too!) to build
this project.

The steps to create this project are:

1. Download 7k historical chess games played by the great Magnus Carlsen
2. Process raw data into an instruction dataset for supervised fine-tuning
3. Fine-tune LFM2-350M to imitate Magnus (aka LFM2-350M-MagnusInstruct)
4. Evaluate the model
5. Bundle the model with `leap-bundle`
6. Create a simple iOS game and embed our fine-tuned LLM using the `LeapSDK` for iOS

Steps 1 to 5 are implemented inside the `fine-tune` directory.
The iOS app is inside `ChessChat`

Let me go over each of these steps, step by step :-)


## 1. Download chess moves dataset

```sh
cd fine-tune && make download-magnus-games
```

We fetch 7k historical games played by the great Magnus Carlsen from [this URL](https://www.pgnmentor.com/files.html#players).

This will download the data to `fine-tune/data/raw/Carlsen.zip` and unzip it to `fine-tune/data/raw/Carlsen.pgn`.
The data is in PGN format, which is a standard format for chess games.

Feel free to expand the dataset to include more games by downloading other players' games from [this URL](https://www.pgnmentor.com/files.html#players).


## 2. Generate instruction dataset

```sh
cd fine-tune && make instruction-dataset
```

For each PGN file we downloaded, we extract

- the game states, 
- the previous 5 moves
- the set of valid moves for the current game state, and
- the next move picked by Magnus.

and store the processed data in a JSON file.

For example, the extracted data for Magnus Carlsen is stored in `fine-tune/data/processed/Carlsen.json`.


## 3. Fine-tune LFM2-350M to imitate Magnus (or any other player you got data for)

```sh
cd fine-tune && make fine-tune
```

We used supervised fine-tuning to train the model to "imitate" Magnus.

The chat template we used is the one used by the LiquidAI team to do general fine-tuning of LFM2-350M, so we don't erase abilities the model had before our fine-tuning.

```jinja2
<|startoftext|><|im_start|>system
You are a helpful assistant trained by Liquid AI.<|im_end|>
<|im_start|>user
...user prompt goes here...<|im_end|>
<|im_start|>assistant
...assistant response goes here...<|im_end|>
```
       
In the user prompt we include
- the game state
- the previous 5 moves, and
- the set of valid moves for the current game state, to help the LLM output a valid move

    ```python
    # see `fine-tune/src/fine_tune/prompt_template.py`
    CHESS_PROMPT_TEMPLATE = """
    You are the great Magnus Carlsen.
    Your task is to make the best move in the given game state.

    Game state: {{ game_state }}
    Last 5 moves: {{ last_5_moves_uci }}
    Valid moves: {{ valid_moves }}

    Your next move should be in UCI format (e.g., 'e2e4', 'f8c8').
    Make sure your next move is one of the valid moves.
    """
    ```

Now, to run the fine-tuning script you NEED at least ONE GPU. So if you don't have one
you need to rent one.

> ___
> **💡 How to get a GPU for LLM training work**
> 
> In my experience, the fastest and most ergonomic way to use a GPU for LLM training work is [Modal](https://modal.com/) serverless platform.
> 
> With Modal you work locally, but your code runs on a remote GPU.
> 
> You define your:
> - training job hardware
> - python dependencies and
> - training logic
> 
> **ALL in Python.** So you don't need to worry about pushing Docker images or switching on an on off pods.
> 
> *This is not an ad. This is just my personal preference.*
> ___

In terms of libraries, we use the standard `SFTTrainer` from the `trl` library, with Unsloth for faster training.

Feel free to play with the configuration parameters in the `fine-tune/src/fine_tune/config.py` file.

### In a nutshell 🌰

I ran training for 10k steps and saw no significant improvement in the model's performance when using `LiquidAI/LFM2-700M` compared to `LiquidAI/LFM2-350M`.
As I deeply care about minimalism, I sticked to the smaller model to ease deployment.

At the end of the day, the model checkpoint I decided to use is [`LFM2-350M-r16-20250902-232247/checkpoint-5000`](https://wandb.ai/paulescu/finetuning-things/runs/vs3s8f1q?nw=nwuserpaulescu), which we can friendly call `LFM2-350M-MagnusInstruct`.

> **💡 Wanna see all the experiments?**
> 
> Click [here](https://wandb.ai/Paulescu/finetuning-things) to see all the experiments I ran and logged to Weights & Biases.


## 4. Evaluate LFM2-350M-MagnusInstruct on real game play

```sh
cd fine-tune && make evaluate
```

### **Warning ⚠️**

It is important to note that the model is not trained to "think" like a chess player.

It is trained to "imitate" Magnus by predicting the next move based on the previous moves and the game state.

And the thing is, chess is a highly-dimenstional state space game.
Which means that to play chess you need way more than memorization.
You need to reason and plan your moves.

An LLM that predicts the next move well in the test (like our `LFM2-350M-MagnusInstruct`) is not necessarily a good chess player.

As a matter of fact, playing random against this LLM is actually a great way to win against it.
Playing random exposes the LLM to a high-number of out-of-sample situation it has never observed during training and the model is not really trained to reason like good chess players do, combining long-term strategic play and short-term tactical play.

Because of this, training a reasoning LLMs using next moves and expert human strategy and tactis
would work better. See [this paper for further reference](https://arxiv.org/html/2411.06655v2).


## 5. Bundle the model with the Leap Model Bundling CLI

```sh
cd fine-tune && make bundle-model
```

The model checkpoints we save during training are the lora weights we need to apply to the base model to get the fine-tuned model.

So, as a first step we need to merge the lora weights with the base model to get the fine-tuned model.
This is what the `merge-model` Makefile target does.

Then, we use the `leap-bundle` CLI to bundle the model into a `.bundle` file.

The result is a `.bundle` file that contains the fine-tuned model and tokenizer, that
can be deployed on mobile devices (both Android and iOS) using the Leap Edge SDK.


## 6. Embed the model into an iOS app with the Leap Edge SDK

To open the app you will need to have Xcode installed.

```sh
cd ChessChat && make open
```

If you want to create this project from scratch, you can follow the steps below.

1. Add the `LeapSDK`, `ChessKit` and `ChessboardKit` packages to the project.
2. Create a `Resources` folder and add the model bundle `LFM2-350M-MagnusInstruct.bundle`.
3. Create a `Models` folder and add the `Player.swift` file.
4. Create a `Views` folder and add the `ContentView.swift` file.

You can find the app in the `ChessChat` directory.
It is a simple app that loads the model and allows you to chat with it.
It uses the `LeapSDK` for iOS to load the model and chat with it.


//You can find the app in the `ChessChat` directory.
It is a simple app that loads the model and allows you to chat with it.
It uses the `LeapSDK` for iOS to load the model and chat with it.

First I decided to bootstrap a minimal chat app that loads the model and allows you to chat with it.
For that I used the `LeapChatExample` from the `LeapSDK-Examples` repository.
You can find it [here](https://leap.liquid.ai/docs/edge-sdk/ios/ios-quick-start-guide#complete-example)

### Steps to create the app

- Create a new project with xcode.
- Create a `Resources` folder and add the model bundle `LFM2-350M-MagnusInstruct.bundle`.
- Add the `LeapSDK` package to the project as explained [here](LFM2-350M-MagnusInstruct).


ChessboardKit for a nice UI:
https://github.com/rohanrhu/ChessboardKit/tree/main?tab=readme-ov-file#quick-start