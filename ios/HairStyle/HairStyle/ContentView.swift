import PhotosUI
import SwiftUI

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var originalImage: UIImage?
    @State private var previewImage: UIImage?
    @State private var preset: Preset = .crown
    @State private var coverage: Double = 50
    @State private var isLoading = false
    @State private var errorMessage: String?

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    PhotosPicker(selection: $selectedItem, matching: .images) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Select Photo")
                                .font(.headline)
                            Text("Choose a clear view of your hair.")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(Color(.secondarySystemBackground))
                        .cornerRadius(12)
                    }
                    .onChange(of: selectedItem) { newItem in
                        Task {
                            await loadImage(from: newItem)
                        }
                    }

                    Picker("Preset", selection: $preset) {
                        ForEach(Preset.allCases) { option in
                            Text(option.rawValue).tag(option)
                        }
                    }
                    .pickerStyle(.segmented)

                    VStack(alignment: .leading, spacing: 8) {
                        Text("Coverage: \(Int(coverage))")
                            .font(.headline)
                        Slider(value: $coverage, in: 0...100, step: 1)
                        HStack {
                            Text("Light")
                            Spacer()
                            Text("Natural")
                            Spacer()
                            Text("Strong")
                        }
                        .font(.caption)
                        .foregroundColor(.secondary)
                    }

                    Button(action: submitPreview) {
                        HStack {
                            Spacer()
                            if isLoading {
                                ProgressView()
                            } else {
                                Text("Preview (Free)")
                                    .font(.headline)
                            }
                            Spacer()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(originalImage == nil || isLoading)

                    if let errorMessage {
                        Text(errorMessage)
                            .foregroundColor(.red)
                    }

                    if let originalImage, let previewImage {
                        BeforeAfterSlider(originalImage: originalImage, previewImage: previewImage)
                    } else if let originalImage {
                        Image(uiImage: originalImage)
                            .resizable()
                            .scaledToFit()
                            .frame(maxWidth: .infinity)
                            .cornerRadius(12)
                    }
                }
                .padding()
            }
            .navigationTitle("HairStyle")
        }
    }

    private func loadImage(from item: PhotosPickerItem?) async {
        previewImage = nil
        errorMessage = nil
        guard let item else {
            originalImage = nil
            return
        }
        do {
            if let data = try await item.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                await MainActor.run {
                    originalImage = image
                }
            }
        } catch {
            await MainActor.run {
                errorMessage = "Unable to load photo."
            }
        }
    }

    private func submitPreview() {
        guard let originalImage else { return }
        isLoading = true
        errorMessage = nil

        Task {
            do {
                let image = try await PreviewService.preview(
                    image: originalImage,
                    preset: preset.rawValue,
                    coverage: Int(coverage)
                )
                await MainActor.run {
                    previewImage = image
                    isLoading = false
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isLoading = false
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
