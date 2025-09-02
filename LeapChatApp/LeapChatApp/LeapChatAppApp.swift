//
//  LeapChatAppApp.swift
//  LeapChatApp
//
//  Created by Pau Labarta Bajo on 2/9/25.
//

import SwiftUI
import LeapSDK
import Observation

@main
struct LeapChatApp: App {
//    @State private var chatStore = ChatStore()
    @State private var chatStore = ChatStore()
 
    var body: some Scene {
        WindowGroup {
            ContentView(chatStore: chatStore)
//                .environment(chatStore)
        }
    }
}
