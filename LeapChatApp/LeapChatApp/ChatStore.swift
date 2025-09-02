//
//  ChatStore.swift
//  LeapChatApp
//
//  Created by Pau Labarta Bajo on 2/9/25.
//

import Foundation
import LeapSDK
import LeapModelDownloader
import SwiftUI

@Observable
class ChatStore {
    // public properties
    var isModelLoading = true
    var isGenerating = false
    var messages: [String] = []
    var error: String?
    
    // private properties
    private var modelRunner: ModelRunner?
    private var conversation: Conversation?
 
    @MainActor
    func setupModel() async {
        do {
            guard let modelURL = Bundle.main.url(
                forResource: "lfm2-350M-8da4w_output_8da8w",
                withExtension: "bundle"
            ) else {
                error = "Could not find model bundle"
                return
            }
 
            modelRunner = try await Leap.load(url: modelURL)
            conversation = Conversation(modelRunner: modelRunner!, history: [])
            isModelLoading = false
        } catch {
            self.error = error.localizedDescription
            isModelLoading = false
        }
    }
 
    @MainActor
    func sendMessage(_ input: String) async {
        guard let conversation = conversation else { return }
 
        isGenerating = true
        let userMessage = ChatMessage(role: .user, content: [.text(input)])
//        messages.append("User: \(input)")
//        messages.append("{\"/moves\": [\"d2d4\", \"g8f6\", \"c2c4\", \"g7g6\", \"b1c3\"]}")
        messages = ["{\"/moves\": [\"d2d4\", \"g8f6\", \"c2c4\", \"g7g6\", \"b1c3\"]}"]
 
        var assistantResponse = ""
        let stream = conversation.generateResponse(message: userMessage)
        
        do {
            for try await response in stream {
                switch response {
                case .chunk(let text):
                    assistantResponse += text
                case .reasoningChunk(_):
                    break // Handle reasoning if needed
                case .complete(_, _):
                    messages.append("Assistant: \(assistantResponse)")
                    isGenerating = false
                case .functionCall(_):
                    break  // Function calls not used in this example
                }
            }
        } catch {
            self.error = error.localizedDescription
            isGenerating = false
        }
    }
}
