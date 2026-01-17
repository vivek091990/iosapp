import SwiftUI

struct BeforeAfterSlider: View {
    let originalImage: UIImage
    let previewImage: UIImage

    @State private var sliderValue: Double = 0.5

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Before / After")
                .font(.headline)
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Image(uiImage: originalImage)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .clipped()

                    Image(uiImage: previewImage)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .clipped()
                        .mask(
                            Rectangle()
                                .frame(width: geometry.size.width * sliderValue)
                        )
                }
            }
            .frame(height: 320)
            .cornerRadius(12)

            Slider(value: $sliderValue, in: 0...1)
        }
    }
}
