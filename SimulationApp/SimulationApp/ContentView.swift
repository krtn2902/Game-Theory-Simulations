//
//  ContentView.swift
//  SimulationApp
//
//  Created by Kirtan Desai on 27/01/25.
//

import SwiftUI
import AppKit

struct ContentView: View {
    let  imageName = "prisoner" // Replace with your actual file name
    let imageExtension = "jpg"
    var body: some View {
        VStack {
            
            VStack {
                if let imageURL = Bundle.main.url(forResource: imageName, withExtension: imageExtension),
                               let nsImage = NSImage(contentsOf: imageURL) {
                                // Convert NSImage to SwiftUI Image
                                Image(nsImage: nsImage)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(width: 200, height: 200)
                            } else {
                                Text("Failed to load image")
                                    .foregroundColor(.red)
                            }
                Text("Iterative Prisoner's Dilemma")
                    .font(.system(size: 16, weight: .bold, design: .serif))
            }
            
            
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
