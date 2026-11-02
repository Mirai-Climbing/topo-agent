import ExecuTorch
import UIKit

/// One-shot climbing-hold segmentation (YOLO11n-seg, ExecuTorch .pte).
/// Mirrors the Android app: letterbox to 640, NMS, per-instance masks.
struct Detection {
  var x1, y1, x2, y2, conf: Float
  var coefs: [Float]
}

final class YoloSeg {
  static let side = 640
  static let proto = 160
  static let coefs = 32
  static let anchors = 8400
  static let confThresh: Float = 0.5
  static let nmsThresh: Float = 0.5
  static let maxDets = 100

  private let module: Module

  init(modelPath: String) throws {
    module = Module(filePath: modelPath)
  }

  /// Letterbox RGB thinkers: longest side to 640, pad 114, NCHW float32 / 255.
  /// Returns the input tensor plus (scale, padX, padY) for mapping boxes back.
  func preprocess(_ image: UIImage) -> (Tensor, Float, Float, Float)? {
    guard let cg = image.cgImage else { return nil }
    let w = CGFloat(cg.width), h = CGFloat(cg.height)
    let scale = CGFloat(YoloSeg.side) / max(w, h)
    let rw = Int(w * scale), rh = Int(h * scale)
    let padX = (YoloSeg.side - rw) / 2, padY = (YoloSeg.side - rh) / 2
    guard let ctx = CGContext(data: nil, width: rw, height: rh,
                              bitsPerComponent: 8, bytesPerRow: rw * 4,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: rw, height: rh))
    guard let px = ctx.data?.assumingMemoryBound(to: UInt8.self) else { return nil }
    let s = YoloSeg.side
    var data = [Float](repeating: 114.0 / 255.0, count: 3 * s * s)
    for y in 0..<rh {
      for x in 0..<rw {
        let o = (y * rw + x) * 4
        let dx = padX + x, dy = padY + y, i = dy * s + dx
        data[i] = Float(px[o]) / 255.0
        data[s * s + i] = Float(px[o + 1]) / 255.0
        data[2 * s * s + i] = Float(px[o + 2]) / 255.0
      }
    }
    let shape: [NSNumber] = [1, 3, NSNumber(value: s), NSNumber(value: s)]
    let t = data.withUnsafeBufferPointer {
      Tensor(data: Data(buffer: $0), shape: shape, dataType: .float)
    }
    return (t, Float(scale), Float(padX), Float(padY))
  }

  func detect(_ image: UIImage) -> (dets: [Detection], proto: [Float], geo: (Float, Float, Float))? {
    guard let (input, scale, padX, padY) = preprocess(image),
          let outputs = try? module.forward([Value(input)]),
          outputs.count >= 2,
          let det = floats(outputs[0]),
          let proto = floats(outputs[1]) else { return nil }
    let w = Float(image.cgImage!.width), h = Float(image.cgImage!.height)
    var cands: [Detection] = []
    for i in 0..<YoloSeg.anchors {
      let conf = det[4 * YoloSeg.anchors + i]
      if conf < YoloSeg.confThresh { continue }
      let cx = det[i], cy = det[YoloSeg.anchors + i]
      let bw = det[2 * YoloSeg.anchors + i], bh = det[3 * YoloSeg.anchors + i]
      let x1 = min(max((cx - bw / 2 - padX) / scale, 0), w)
      let y1 = min(max((cy - bh / 2 - padY) / scale, 0), h)
      let x2 = min(max((cx + bw / 2 - padX) / scale, 0), w)
      let y2 = min(max((cy + bh / 2 - padY) / scale, 0), h)
      if x2 - x1 < 2 || y2 - y1 < 2 { continue }
      var coefs = [Float](repeating: 0, count: YoloSeg.coefs)
      for k in 0..<YoloSeg.coefs { coefs[k] = det[(5 + k) * YoloSeg.anchors + i] }
      cands.append(Detection(x1: x1, y1: y1, x2: x2, y2: y2, conf: conf, coefs: coefs))
    }
    return (nms(cands), proto, (scale, padX, padY))
  }

  /// Copies a float tensor output into an array (ExecuTorch 0.6 has no toTensor()/data()).
  private func floats(_ value: Value) -> [Float]? {
    guard let tensor = value.tensor else { return nil }
    let n = tensor.shape.reduce(1) { $0 * $1.intValue }
    var out: [Float]?
    tensor.bytes { pointer, _, dataType in
      guard dataType == .float else { return }
      out = Array(UnsafeBufferPointer(start: pointer.assumingMemoryBound(to: Float.self), count: n))
    }
    return out
  }

  private func nms(_ boxes: [Detection]) -> [Detection] {
    var sorted = boxes.sorted { $0.conf > $1.conf }
    var kept: [Detection] = []
    while !sorted.isEmpty && kept.count < YoloSeg.maxDets {
      let a = sorted.removeFirst()
      kept.append(a)
      sorted.removeAll { b in
        let iw = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
        let ih = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
        let inter = iw * ih
        let union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return union > 0 && inter / union > YoloSeg.nmsThresh
      }
    }
    return kept
  }

  /// Tint every mask (sigmoid mask at proto res, bilinear-upsampled to the box).
  func render(base: UIImage, dets: [Detection], proto: [Float],
              geo: (Float, Float, Float), selected: Int, palette: [UIColor]) -> UIImage? {
    guard let cg = base.cgImage else { return nil }
    let w = cg.width, h = cg.height
    let (scale, padX, padY) = geo
    guard let ctx = CGContext(data: nil, width: w, height: h,
                              bitsPerComponent: 8, bytesPerRow: w * 4,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
          let buf = ctx.data?.assumingMemoryBound(to: UInt8.self) else { return nil }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    let p = YoloSeg.proto, n = YoloSeg.coefs
    let order = dets.indices.sorted { ($0 == selected ? 1 : 0) < ($1 == selected ? 1 : 0) }
    for idx in order {
      let d = dets[idx]
      let bright = idx == selected
      let alpha: Float = bright ? 0.7 : 0.45
      var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0
      palette[idx % palette.count].getRed(&r, green: &g, blue: &b, alpha: nil)
      let cr = Float(r * 255) * alpha, cg2 = Float(g * 255) * alpha, cb = Float(b * 255) * alpha
      let gx0 = (d.x1 * scale + padX) / 4, gy0 = (d.y1 * scale + padY) / 4
      let gx1 = (d.x2 * scale + padX) / 4, gy1 = (d.y2 * scale + padY) / 4
      let ix0 = min(max(Int(gx0) - 1, 0), p - 1), iy0 = min(max(Int(gy0) - 1, 0), p - 1)
      let ix1 = min(max(Int(gx1) + 1, 0), p - 1), iy1 = min(max(Int(gy1) + 1, 0), p - 1)
      let mw = ix1 - ix0 + 1, mh = iy1 - iy0 + 1
      var small = [Float](repeating: 0, count: mw * mh)
      for cy in 0..<mh {
        for cx in 0..<mw {
          let q = (iy0 + cy) * p + (ix0 + cx)
          var v: Float = 0
          for k in 0..<n { v += d.coefs[k] * proto[k * p * p + q] }
          small[cy * mw + cx] = 1 / (1 + exp(-v))
        }
      }
      let bx0 = min(max(Int(d.x1), 0), w - 1), by0 = min(max(Int(d.y1), 0), h - 1)
      let bx1 = min(max(Int(d.x2), 0), w), by1 = min(max(Int(d.y2), 0), h)
      for y in by0..<by1 {
        let fy = (Float(y) + 0.5) * scale + padY
        let fy4 = fy / 4 - Float(iy0)
        let y0 = min(max(Int(fy4), 0), mh - 1), y1 = min(y0 + 1, mh - 1)
        let wy = min(max(fy4 - floor(fy4), 0), 1)
        for x in bx0..<bx1 {
          let fx = (Float(x) + 0.5) * scale + padX
          let fx4 = fx / 4 - Float(ix0)
          let x0 = min(max(Int(fx4), 0), mw - 1), x1 = min(x0 + 1, mw - 1)
          let wx = min(max(fx4 - floor(fx4), 0), 1)
          let v = small[y0 * mw + x0] * (1 - wx) * (1 - wy)
            + small[y0 * mw + x1] * wx * (1 - wy)
            + small[y1 * mw + x0] * (1 - wx) * wy
            + small[y1 * mw + x1] * wx * wy
          if v > 0.5 {
            let o = (y * w + x) * 4
            buf[o] = UInt8(min(max(Float(buf[o]) * (1 - alpha) + cr, 0), 255))
            buf[o + 1] = UInt8(min(max(Float(buf[o + 1]) * (1 - alpha) + cg2, 0), 255))
            buf[o + 2] = UInt8(min(max(Float(buf[o + 2]) * (1 - alpha) + cb, 0), 255))
          }
        }
      }
    }
    guard let out = ctx.makeImage() else { return nil }
    return UIImage(cgImage: out)
  }
}
