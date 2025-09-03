//
//  ChessChatApp.swift
//  ChessChat
//
//  Created by Pau Labarta Bajo on 3/9/25.
//

import SwiftUI
import LeapSDK
import Observation

@main
struct LeapChatApp: App {
    @State private var llmPlayer = LLMPlayer()
 
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(llmPlayer)
        }
    }
}
