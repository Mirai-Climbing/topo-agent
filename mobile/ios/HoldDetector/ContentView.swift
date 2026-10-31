import PhotosUI
import SwiftUI

struct ContentView: View {
  @State private var photo: UIImage?
  @State private var overlay: UIImage?
  @State private var showOverlay = true
  @State private var status = "Pick a photo or take one"
  @State private var pickerItem: PhotosPickerItem?
  @State private var showCamera = false
  @State private var dets: [Detection] = []
  @State private var proto: [Float] = []
  @State private var geo: (Float, Float, Float) = (1, 0, 0)
  @State private var selected = -1
  @State private var engine: YoloSeg?

  let palette: [UIColor] = [
    .systemCyan, .systemGreen, .systemOrange, .systemPurple,
    .systemPink, .systemIndigo, .systemTeal, .systemRed,
  ]

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text("Hold detector").font(.title).bold()
      Text("YOLO11s-seg · one pass, every hold · tap to inspect")
        .font(.caption).foregroundColor(.secondary)
      Text(status).font(.headline).foregroundColor(.green)
      HStack {
        PhotosPicker(selection: $pickerItem, matching: .images) {
          Text("Photo library")
        }.buttonStyle(.borderedProminent)
        Button("Camera") { showCamera = true }.buttonStyle(.borderedProminent)
        Button(showOverlay ? "Show original" : "Show detection") {
          showOverlay.toggle()
        }.buttonStyle(.bordered)
      }
      if let img = showOverlay ? (overlay ?? photo) : photo {
        Image(uiImage: img).resizable().scaledToFit()
          .gesture(DragGesture(minimumDistance: 0).onEnded { g in
            tap(at: g.location, in: img)
          })
      } else {
        Spacer()
      }
    }
    .padding()
    .task { setup() }
    .onChange(of: pickerItem) { _, item in
      guard let item else { return }
      Task {
        if let data = try? await item.loadTransferable(type: Data.self),
           let ui = UIImage(data: data) { run(ui) }
      }
    }
    .sheet(isPresented: $showCamera) {
      CameraView { ui in run(ui) }
    }
  }

  private func setup() {
    guard engine == nil,
          let url = Bundle.main.url(forResource: "hold_yolo", withExtension: "pte") else {
      status = "hold_yolo.pte missing from bundle"
      return
    }
    do {
      engine = try YoloSeg(modelPath: url.path)
    } catch {
      status = "model load failed: \(error)"
    }
  }

  private func run(_ ui: UIImage) {
    guard let engine else { return }
    status = "detecting holds…"
    selected = -1
    dets = []
    overlay = nil
    photo = ui
    DispatchQueue.global(qos: .userInitiated).async {
      let t0 = Date()
      guard let (found, pr, g) = engine.detect(ui) else {
        DispatchQueue.main.async { status = "detection failed" }
        return
      }
      let ms = Date().timeIntervalSince(t0) * 1000
      let ov = engine.render(base: ui, dets: found, proto: pr, geo: g,
                             selected: -1, palette: palette)
      DispatchQueue.main.async {
        dets = found
        proto = pr
        geo = g
        overlay = ov
        showOverlay = true
        status = String(format: "hold-yolo · %d holds · infer %.0f ms", found.count, ms)
      }
    }
  }

  private func tap(at point: CGPoint, in img: UIImage) {
    // Map the tap through AspectFit into image pixels (same idea as Android).
    // Simplified: assumes the image fills the view width.
    guard !dets.isEmpty, !proto.isEmpty else { return }
    var best = -1, bestArea = Float.greatestFiniteMagnitude
    let s = img.size.width / UIScreen.main.bounds.width
    let x = Float(point.x) * Float(s), y = Float(point.y) * Float(s)
    for (i, d) in dets.enumerated() {
      if x >= d.x1 && x <= d.x2 && y >= d.y1 && y <= d.y2 {
        let a = (d.x2 - d.x1) * (d.y2 - d.y1)
        if a < bestArea { bestArea = a; best = i }
      }
    }
    guard best >= 0, let engine, let ui = photo,
          let ov = engine.render(base: ui, dets: dets, proto: proto, geo: geo,
                                 selected: best, palette: palette) else { return }
    selected = best
    overlay = ov
    status = String(format: "hold #%d · conf %.2f · %d holds", best, dets[best].conf, dets.count)
  }
}

struct CameraView: UIViewControllerRepresentable {
  var onCapture: (UIImage) -> Void
  func makeUIViewController(context: Context) -> UIImagePickerController {
    let c = UIImagePickerController()
    c.sourceType = .camera
    c.delegate = context.coordinator
    return c
  }
  func updateUIViewController(_ ui: UIImagePickerController, context: Context) {}
  func makeCoordinator() -> Coord { Coord(onCapture: onCapture) }
  class Coord: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    var onCapture: (UIImage) -> Void
    init(onCapture: @escaping (UIImage) -> Void) { self.onCapture = onCapture }
    func imagePickerController(_ p: UIImagePickerController,
                               didFinishPickingMediaWithInfo i: [UIImagePickerController.InfoKey: Any]) {
      p.dismiss(animated: true)
      if let ui = i[.originalImage] as? UIImage { onCapture(ui) }
    }
  }
}
