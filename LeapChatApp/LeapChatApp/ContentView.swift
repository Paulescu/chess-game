//
//  ContentView.swift
//  LeapChatApp
//
//  Created by Pau Labarta Bajo on 2/9/25.
//

import SwiftUI

struct ContentView: View {
//    @Environment(ChatStore.self) private var chatStore
    let chatStore: ChatStore
    @State private var inputText = ""
 
    var body: some View {
        VStack {
            Text("I am alive!")
            if chatStore.isModelLoading {
                ProgressView("Loading model...")
                    .task {
                        await chatStore.setupModel()
                    }
            } else if let error = chatStore.error {
                Text("Error: \(error)")
                    .foregroundColor(.red)
            } else {
                // Message list
                ScrollView {
                    ForEach(chatStore.messages, id: \.self) { message in
                        Text(message)
                            .padding()
                    }
                }
 
                // Input field
                HStack {
                    TextField("Type a message", text: $inputText)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
 
                    Button("Send") {
                        Task {
                            await chatStore.sendMessage(inputText)
                            inputText = ""
                        }
                    }
                    .disabled(chatStore.isGenerating || inputText.isEmpty)
                }
                .padding()
            }
        }
    }
}

#Preview {
    ContentView(chatStore: ChatStore())
}
