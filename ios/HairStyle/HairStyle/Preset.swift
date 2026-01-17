import Foundation

enum Preset: String, CaseIterable, Identifiable {
    case crown = "Crown"
    case hairline = "Hairline"
    case part = "Part"

    var id: String { rawValue }
}
