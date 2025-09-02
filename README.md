## Let's build a Chess game using LLMs running privately on your phone/computer/microwave

## TODOs

- [x] Fine-tune Gemma-3-270M using the ChessInstruct dataset using Modal
- [x] Fine-tune LFM2-350M using the ChessInstruct dataset using Modal
- [x] Evaluate the model.
- [x] Bundle the model with leap-bundle
- [ ] Deploy model to minimal iOS app.


- [ ] Build a basic chat app from scratch using [this example](https://leap.liquid.ai/docs/edge-sdk/ios/ios-quick-start-guide#complete-example) from the LeapSDK page

- [ ] Build iOS app where user types in a text field the sequence of previous moves, clicks on a button, and the app throws out the predicted next_move.


Steps:

1. Create boilerplate SwiftUI app with xcode.
2. Import the LeapSDK into the project
2. Add model bundle to the project, preferably under a dedicated `Resources/` dir to keep the project file well organized.

