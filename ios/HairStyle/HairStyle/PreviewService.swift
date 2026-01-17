import Foundation
import UIKit

enum PreviewService {
    enum PreviewError: LocalizedError {
        case encodingFailed
        case invalidResponse

        var errorDescription: String? {
            switch self {
            case .encodingFailed:
                return "Unable to encode image."
            case .invalidResponse:
                return "Unexpected response from server."
            }
        }
    }

    static func preview(image: UIImage, preset: String, coverage: Int) async throws -> UIImage {
        guard let url = URL(string: "/v1/preview", relativeTo: Config.apiBaseURL) else {
            throw PreviewError.invalidResponse
        }
        guard let imageData = image.pngData() else {
            throw PreviewError.encodingFailed
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        let body = MultipartFormDataBuilder(boundary: boundary)
            .addField(named: "preset", value: preset)
            .addField(named: "coverage", value: String(coverage))
            .addFile(named: "file", filename: "photo.png", mimeType: "image/png", data: imageData)
            .build()

        request.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw PreviewError.invalidResponse
        }
        guard let responseImage = UIImage(data: data) else {
            throw PreviewError.invalidResponse
        }
        return responseImage
    }
}

struct MultipartFormDataBuilder {
    let boundary: String
    private var data = Data()

    init(boundary: String) {
        self.boundary = boundary
    }

    mutating func addField(named name: String, value: String) -> Self {
        let fieldData = "--\(boundary)\r\n" +
            "Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n" +
            "\(value)\r\n"
        data.append(Data(fieldData.utf8))
        return self
    }

    mutating func addFile(named name: String, filename: String, mimeType: String, data fileData: Data) -> Self {
        let header = "--\(boundary)\r\n" +
            "Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n" +
            "Content-Type: \(mimeType)\r\n\r\n"
        data.append(Data(header.utf8))
        data.append(fileData)
        data.append(Data("\r\n".utf8))
        return self
    }

    func build() -> Data {
        var finalData = data
        finalData.append(Data("--\(boundary)--\r\n".utf8))
        return finalData
    }
}
