import SwiftUI

enum Move: String {
    case C = "Cooperate"
    case D = "Defect"
}

struct Strategy: Identifiable {
    let id = UUID()
    let name: String
    let strategy: ([Move], [Move]) -> Move
}

struct Matchup: Identifiable {
    let id = UUID()
    let strategy1: String
    let strategy2: String
    var player1Moves: [Move]
    var player2Moves: [Move]
    var score1: Int
    var score2: Int
}

struct ContentView: View {
    let payoffMatrix: [Move: [Move: Int]] = [
        .C: [.C: 3, .D: 0],
        .D: [.C: 5, .D: 1]
    ]
    
    let strategies: [Strategy] = [
        Strategy(name: "Tit for Tat") { own, opponent in
            opponent.last ?? .C
        },
        Strategy(name: "Always Cooperate") { _, _ in .C },
        Strategy(name: "Always Defect") { _, _ in .D },
        Strategy(name: "Tit for Two Tats") { own, opponent in
            opponent.suffix(2) == [.D, .D] ? .D : .C
        },
        Strategy(name: "Grudger") { own, opponent in
            opponent.contains(.D) ? .D : .C
        },
        Strategy(name: "Random") { _, _ in
            [Move.C, .D].randomElement()!
        },
        Strategy(name: "Pavlov") { own, opponent in
            guard !own.isEmpty else { return .C }
            let lastOwn = own.last!
            let lastOpponent = opponent.last ?? .C
            return lastOwn == lastOpponent ? lastOwn : (lastOwn == .C ? .D : .C)
        }
    ]
    
    @State private var matchups: [Matchup] = []
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Payoff Matrix Display
                VStack(alignment: .leading, spacing: 10) {
                    
                    
                    Text("Payoff Matrix")
                        .font(.headline.bold())
                    
                    Grid(alignment: .leading) {
                        GridRow {
                            Text("")
                            Text("Cooperate").bold()
                            Text("Defect").bold()
                        }
                        
                        GridRow {
                            Text("Cooperate").bold()
                            Text("3 / 3")
                            Text("0 / 5")
                        }
                        
                        GridRow {
                            Text("Defect").bold()
                            Text("5 / 0")
                            Text("1 / 1")
                        }
                    }
                    .font(.system(size: 14))
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(10)
                
                Button("Run All Matchups") {
                    runAllMatchups()
                }
                .buttonStyle(.borderedProminent)
                .padding(.vertical)
                
                ForEach(matchups) { matchup in
                    VStack(spacing: 20) {
                        Text("\(matchup.strategy1) vs \(matchup.strategy2)")
                            .font(.title3.bold())
                        
                        // Player 1 Section
                        VStack(spacing: 10) {
                            Text(matchup.strategy1)
                                .font(.headline)
                            movesView(moves: matchup.player1Moves)
                            Text("Score: \(matchup.score1)")
                                .font(.subheadline.bold())
                        }
                        .padding()
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(10)
                        
                        // Player 2 Section
                        VStack(spacing: 10) {
                            Text(matchup.strategy2)
                                .font(.headline)
                            movesView(moves: matchup.player2Moves)
                            Text("Score: \(matchup.score2)")
                                .font(.subheadline.bold())
                        }
                        .padding()
                        .background(Color.green.opacity(0.1))
                        .cornerRadius(10)
                        
                        Divider()
                            .padding(.vertical)
                    }
                    .padding(.horizontal)
                }
            }
            .padding()
        }
        .frame(minWidth: 1000, minHeight: 800)
    }
    
    func movesView(moves: [Move]) -> some View {
        HStack {
            ForEach(0..<moves.count, id: \.self) { index in
                VStack(spacing: 4) {
                    Circle()
                        .fill(moves[index] == .D ? Color.red : Color.green)
                        .frame(width: 30, height: 30)
                        .overlay(
                            Text("\(index + 1)")
                                .font(.system(size: 10, weight: .bold))
                                .foregroundColor(.white)
                        )
                    
                    Text(moves[index].rawValue.prefix(1))
                        .font(.caption2)
                }
            }
        }
    }
    
    func runAllMatchups() {
        let titForTat = strategies.first!
        var newMatchups: [Matchup] = []
        
        for strategy in strategies {
            let (s1m, s2m, s1, s2) = simulateMatch(
                strat1: titForTat.strategy,
                strat2: strategy.strategy
            )
            
            let matchup = Matchup(
                strategy1: titForTat.name,
                strategy2: strategy.name,
                player1Moves: s1m,
                player2Moves: s2m,
                score1: s1,
                score2: s2
            )
            
            newMatchups.append(matchup)
        }
        
        matchups = newMatchups
    }
    
    func simulateMatch(strat1: @escaping ([Move], [Move]) -> Move,
                       strat2: @escaping ([Move], [Move]) -> Move) -> ([Move], [Move], Int, Int) {
        var history1: [Move] = []
        var history2: [Move] = []
        var score1 = 0
        var score2 = 0
        
        for _ in 0..<10 {
            let move1 = strat1(history1, history2)
            let move2 = strat2(history2, history1)
            
            score1 += payoffMatrix[move1]![move2]!
            score2 += payoffMatrix[move2]![move1]!
            
            history1.append(move1)
            history2.append(move2)
        }
        
        return (history1, history2, score1, score2)
    }
}

#Preview {
    ContentView()
}
