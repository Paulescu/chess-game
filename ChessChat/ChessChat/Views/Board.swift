//
//  Board.swift
//  ChessChat
//
//  Created by Pau Labarta Bajo on 3/9/25.
//
import SwiftUI
import ChessboardKit
import ChessKit

public struct Board: View {
    static let POSITION = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    @Environment(LLMPlayer.self) private var llmPlayer
    
    @State var showError: Bool = false
    @State var errorMessage: String = ""
    @State var size: CGFloat = 350
    @State var fen = POSITION
    
    @Bindable var chessboardModel = ChessboardModel(fen: POSITION,
                                                    perspective: .white,
                                                    colorScheme: .light)
    
    @State var moves = [Move]()
    @State var isAIThinking = false
    
    var backgroundAnimationStartDate = Date()
    
    public var body: some View {
        VStack(spacing: 20) {
            VStack {}.frame(height: 50)
            
            Text("Play against MagnusInstruct")
                .font(.title)
                .fontWeight(.bold)
                .multilineTextAlignment(.center)
            
            Chessboard(chessboardModel: chessboardModel)
                .onMove { move, isLegal, from, to, lan, promotionPiece in
                    print("Move: Fen: \(chessboardModel.fen) - Lan: \(lan)")
                    
                    if !isLegal {
                        print("Illegal move: \(lan)")
                        return
                    }
                    
                    chessboardModel.game.make(move: move)
                    chessboardModel.setFen(FenSerialization.default.serialize(position: chessboardModel.game.position), lan: lan)
                    
                    print("Move: \(lan)")
                    moves.append(move)
                    
                    // Now it is the LLMs turn to play
                    Task {
                         await getAIMove()
                    }
                }
                .frame(width: size, height: size)
                .padding(5)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.gray, lineWidth: 1))
            
            Spacer()
            
            if showError {
                Text(errorMessage)
                    .foregroundColor(.red)
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }
            
            Button("Restart Game") {
                chessboardModel.setFen(Self.POSITION)
                moves.removeAll()
                fen = Self.POSITION
            }
            .font(.headline)
            .foregroundColor(.white)
            .frame(width: 200, height: 50)
            .background(
                LinearGradient(
                    gradient: Gradient(colors: [Color.blue, Color.blue.opacity(0.8)]),
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
            .cornerRadius(12)
            .shadow(color: .black.opacity(0.2), radius: 4, x: 0, y: 2)
        }
        .padding()
        .background {
            GeometryReader { proxy in
                ZStack {
                    TimelineView(.animation) { context in
                        Color.white
                            .scaledToFill()
                            .visualEffect { content, proxy in
                                content
                                    .colorEffect(ShaderLibrary.circlesBackground(
                                        .boundingRect,
                                        .float(backgroundAnimationStartDate.timeIntervalSinceNow),
                                        .color(Color(hue: 0.0, saturation: 0.0, brightness: 0.935)),
                                        .color(Color(hue: 0.0, saturation: 0.0, brightness: 0.890))
                                    ))
                            }
                    }
                    
                    LinearGradient(
                        stops: [
                            .init(color: .white.opacity(0.1), location: 0),
                            .init(color: .white.opacity(0.9), location: 0.33)
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                }
                .frame(width: proxy.size.width, height: proxy.size.height)
                .clipped()
            }
        }
    }
    
    func getAIMove() async {
        isAIThinking = true
        
        while true {
            do {
                let move_str = try await llmPlayer.getNextMove(game: chessboardModel.game)
                let move = Move(string: move_str)
                
                if !chessboardModel.game.legalMoves.contains(move) {
                    throw ChessError.invalidMove(move_str)
                }
                
                print("AI wants to play \(move_str)")
                
                // Apply the AI move to the board
                chessboardModel.game.make(move: move)
                chessboardModel.setFen(FenSerialization.default.serialize(position: chessboardModel.game.position))
                
                showError = false
                break
                
            } catch {
                errorMessage = "AI move failed:\(error.localizedDescription)"
                showError = true
            }
            isAIThinking = false
        }
    }
}
