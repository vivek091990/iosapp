import Foundation
import UIKit

enum PreviewService {
    enum PreviewError: LocalizedError {
        case encodingFailed
        case invalidResponse
        case server(String)

        var errorDescription: String? {
            switch self {
            case .encodingFailed:
                return "Unable to prepare the photo."
            case .invalidResponse:
                return "Something went wrong. Please try again."
            case .server(let message):
                return message
            }
        }
    }

    /// Longest side sent to the server. The backend works at about 1.5 MP, so more is wasted upload.
    static let maxUploadSide: CGFloat = 1600

    static func preview(image: UIImage, preset: String, coverage: Int) async throws -> UIImage {
        guard let url = URL(string: "/v1/preview", relativeTo: Config.apiBaseURL) else {
            throw PreviewError.invalidResponse
        }
        guard let imageData = image.resizedForUpload(maxSide: maxUploadSide).jpegData(compressionQuality: 0.9) else {
            throw PreviewError.encodingFailed
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 90

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var builder = MultipartFormDataBuilder(boundary: boundary)
        builder.addField(named: "preset", value: preset)
        builder.addField(named: "coverage", value: String(coverage))
        builder.addFile(named: "file", filename: "photo.jpg", mimeType: "image/jpeg", data: imageData)
        request.httpBody = builder.build()

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw PreviewError.invalidResponse
        }
        guard (200...299).contains(httpResponse.statusCode) else {
            throw PreviewError.server(serverMessage(from: data) ?? "Something went wrong (\(httpResponse.statusCode)). Please try again.")
        }
        guard let responseImage = UIImage(data: data) else {
            throw PreviewError.invalidResponse
        }
        return responseImage
    }

    /// FastAPI errors look like {"detail": "..."}; the backend writes those for people to read.
    private static func serverMessage(from data: Data) -> String? {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let detail = object["detail"] as? String,
              !detail.isEmpty else {
            return nil
        }
        return detail
    }
}

extension UIImage {
    /// Redraws the image upright (applying its orientation) and no larger than `maxSide`.
    func resizedForUpload(maxSide: CGFloat) -> UIImage {
        let longest = max(size.width, size.height)
        let scale = longest > maxSide ? maxSide / longest : 1
        let target = CGSize(width: (size.width * scale).rounded(), height: (size.height * scale).rounded())
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        return UIGraphicsImageRenderer(size: target, format: format).image { _ in
            draw(in: CGRect(origin: .zero, size: target))
        }
    }
}

struct MultipartFormDataBuilder {
    let boundary: String
    private var data = Data()

    init(boundary: String) {
        self.boundary = boundary
    }

    mutating func addField(named name: String, value: String) {
        let fieldData = "--\(boundary)\r\n" +
            "Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n" +
            "\(value)\r\n"
        data.append(Data(fieldData.utf8))
    }

    mutating func addFile(named name: String, filename: String, mimeType: String, data fileData: Data) {
        let header = "--\(boundary)\r\n" +
            "Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n" +
            "Content-Type: \(mimeType)\r\n\r\n"
        data.append(Data(header.utf8))
        data.append(fileData)
        data.append(Data("\r\n".utf8))
    }

    func build() -> Data {
        var finalData = data
        finalData.append(Data("--\(boundary)--\r\n".utf8))
        return finalData
    }
}
