//
//  ContentView.swift
//  ChessChat
//
//  Created by Pau Labarta Bajo on 3/9/25.
//

import SwiftUI
import ChessboardKit
import ChessKit

struct ContentView: View {
    @Environment(LLMPlayer.self) private var llmPlayer
    
    var body: some View {
        VStack {
            if !llmPlayer.isReady {
                ProgressView("Loading the LLM...")
                    .task {
                        await llmPlayer.setupLLM()
                    }
            } else {
                Board()
            }
        }
    }
}

#Preview {
    ContentView()
        .environment(LLMPlayer())
}
