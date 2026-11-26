import Foundation

enum Preset: String, CaseIterable, Identifiable {
    case auto = "Auto"
    case hairline = "Hairline"
    case crown = "Crown"
    case part = "Part"

    var id: String { rawValue }
}
