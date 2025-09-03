import Foundation
import LeapSDK
import ChessKit

@Observable
class LLMPlayer {
    
    var isReady: Bool = false
    var error: String?
    private var modelRunner: ModelRunner?
    private var conversation: Conversation?
    
    func setupLLM() async {
        do {
            guard let modelURL = Bundle.main.url(
                forResource: "LFM2-350M-MagnusInstruct",
                withExtension: "bundle"
            ) else {
                error = "Could not find model bundle"
                return
            }

            modelRunner = try await Leap.load(url: modelURL)
            conversation = Conversation(modelRunner: modelRunner!, history: [])
            isReady = true

        } catch {
            self.error = error.localizedDescription
            isReady = true
        }
    }
    
    func getNextMove(game: Game) async throws -> String {
        guard let conversation = conversation else {
            throw ChessError.modelNotInitialized
        }
        
        let gameState = FenSerialization.default.serialize(position: game.position)
        let last5MovesUci = game.movesHistory.suffix(5).map(\.description)
        let validMoves = game.legalMoves.map(\.description)
        print("validMoves: \(validMoves)")
        
        let prompt = getPrompt(
            gameState: gameState,
            last5MovesUci: last5MovesUci,
            validMoves: validMoves
        )
        
        let userMessage = ChatMessage(role: .user, content: [.text(prompt)])
        
        var options = GenerationOptions()
        options.temperature = 0.8
        
        let stream = conversation.generateResponse(
            message: userMessage,
            generationOptions: options,
        )
        
        var assistantResponse = ""
        
        for try await response in stream {
            switch response {
            case .chunk(let text):
                assistantResponse += text
            case .reasoningChunk(_):
                break
            case .complete(_, _):
                return assistantResponse.trimmingCharacters(in: .whitespacesAndNewlines)
            case .functionCall(_):
                break
            }
        }
        
        return assistantResponse.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    // MARK: private part of the class
    
    private func getPrompt(gameState: String, last5MovesUci: [String], validMoves: [String]) -> String {
        return """
    You are the great Magnus Carlsen.
    Your task is to make the best move in the given game state.

    Game state: \(gameState)
    Last 5 moves: \(last5MovesUci)
    Valid moves: \(validMoves)

    Your next move should be in UCI format (e.g., 'e2e4', 'f8c8').
    Make sure your next move is one of the valid moves.
    """
    }
}

enum ChessError: Error {
    case invalidMove(String)
    case illegalMove(String)
    case noValidMoves
    case modelNotFound
    case modelNotInitialized
}
