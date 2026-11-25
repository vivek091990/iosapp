import PhotosUI
import SwiftUI

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var originalImage: UIImage?
    @State private var previewImage: UIImage?
    @State private var preset: Preset = .auto
    @State private var coverage: Double = 50
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var showConsent = false
    @State private var requestID = UUID()

    /// Photos go to a third-party AI service; ask once before the first upload.
    @AppStorage("hasAcceptedAIProcessing") private var hasAcceptedAIProcessing = false

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    PhotosPicker(selection: $selectedItem, matching: .images) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(originalImage == nil ? "Choose a Photo" : "Choose Another Photo")
                                .font(.headline)
                            Text("A clear, front-facing photo with your hairline visible works best.")
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

                    if isLoading {
                        HStack(spacing: 12) {
                            ProgressView()
                            Text("Filling in your hairline. This takes a few seconds.")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }

                    if let errorMessage {
                        Text(errorMessage)
                            .foregroundColor(.red)
                    }

                    if let originalImage, let previewImage {
                        BeforeAfterSlider(originalImage: originalImage, previewImage: previewImage)
                        ShareLink(
                            item: Image(uiImage: previewImage),
                            preview: SharePreview("My photo", image: Image(uiImage: previewImage))
                        ) {
                            Label("Save or Share", systemImage: "square.and.arrow.up")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                    } else if let originalImage {
                        Image(uiImage: originalImage)
                            .resizable()
                            .scaledToFit()
                            .frame(maxWidth: .infinity)
                            .cornerRadius(12)
                    }

                    if originalImage != nil {
                        adjustments
                    }
                }
                .padding()
            }
            .navigationTitle("HairStyle")
            .alert("Process this photo with AI?", isPresented: $showConsent) {
                Button("Continue") {
                    hasAcceptedAIProcessing = true
                    submitPreview()
                }
                Button("Not Now", role: .cancel) {}
            } message: {
                Text("Your photo is sent to our server and an AI image service to create the preview. It isn't stored after the preview is made.")
            }
        }
    }

    private var adjustments: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Adjust")
                .font(.headline)

            Picker("Area", selection: $preset) {
                ForEach(Preset.allCases) { option in
                    Text(option.rawValue).tag(option)
                }
            }
            .pickerStyle(.segmented)

            VStack(alignment: .leading, spacing: 8) {
                Slider(value: $coverage, in: 0...100, step: 1)
                HStack {
                    Text("Light")
                    Spacer()
                    Text("Natural")
                    Spacer()
                    Text("Full")
                }
                .font(.caption)
                .foregroundColor(.secondary)
            }

            Button(action: requestPreview) {
                Text(previewImage == nil ? "Preview" : "Update Preview")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(isLoading)
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
    }

    private func loadImage(from item: PhotosPickerItem?) async {
        await MainActor.run {
            previewImage = nil
            errorMessage = nil
        }
        guard let item else {
            await MainActor.run { originalImage = nil }
            return
        }
        do {
            if let data = try await item.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                await MainActor.run {
                    originalImage = image
                    // One tap: picking a photo starts the preview with the current settings.
                    requestPreview()
                }
            } else {
                await MainActor.run { errorMessage = "Unable to load photo." }
            }
        } catch {
            await MainActor.run {
                errorMessage = "Unable to load photo."
            }
        }
    }

    private func requestPreview() {
        if hasAcceptedAIProcessing {
            submitPreview()
        } else {
            showConsent = true
        }
    }

    private func submitPreview() {
        guard let originalImage else { return }
        let id = UUID()
        requestID = id
        isLoading = true
        errorMessage = nil
        let presetName = preset.rawValue
        let coverageValue = Int(coverage)

        Task {
            do {
                let image = try await PreviewService.preview(image: originalImage, preset: presetName, coverage: coverageValue)
                await MainActor.run {
                    guard requestID == id else { return }  // a newer photo or setting won
                    previewImage = image
                    isLoading = false
                }
            } catch {
                await MainActor.run {
                    guard requestID == id else { return }
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
